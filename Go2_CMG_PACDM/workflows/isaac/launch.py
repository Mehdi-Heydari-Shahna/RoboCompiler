#!/usr/bin/env python3
"""Launch isolated Isaac processes from the active Miniforge controller Python.

The launcher itself uses only Python's standard library. It never imports Isaac
or Pinocchio into its own process. Result folders are unique and never reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
FULL_CASES = ("nominal", "fine", "low_friction", "payload", "strong_push", "no_actuation", "PD_ablation")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _inside(path, root):
    try:
        path = os.path.normcase(os.path.abspath(path))
        root = os.path.normcase(os.path.abspath(root))
        return os.path.commonpath((path, root)) == root
    except (OSError, ValueError):
        return False


def isaac_environment(source, runtime_python=None):
    """Remove controller-env influence only in the child; retain OS/GPU config."""
    env = dict(source)
    prefixes = [value for key, value in env.items()
                if (key == "CONDA_PREFIX" or key.startswith("CONDA_PREFIX_")) and value]
    if env.get("VIRTUAL_ENV"):
        prefixes.append(env["VIRTUAL_ENV"])
    for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        if key in env:
            parts = [p for p in env[key].split(os.pathsep)
                     if p and not any(_inside(p, prefix) for prefix in prefixes)]
            env[key] = os.pathsep.join(parts)
    for key in list(env):
        if key.startswith("CONDA_") or key in {
            "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
            "VIRTUAL_ENV", "_CE_CONDA", "_CE_M", "PYTHONEXECUTABLE",
        }:
            env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # A separately installed pip runtime may itself live in a Conda env.
    if runtime_python:
        parent = Path(runtime_python).parent
        paths = [str(parent)]
        if os.name == "nt":
            paths += [str(parent / "Scripts"), str(parent / "Library" / "bin")]
        env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    return env


def controller_probe():
    script = """import json, sys
import numpy, scipy, pinocchio, osqp, imageio_ffmpeg, matplotlib
print(json.dumps(dict(python=sys.executable, python_version=sys.version,
    versions={m.__name__: getattr(m, '__version__', 'unknown') for m in
              (numpy, scipy, pinocchio, osqp, imageio_ffmpeg, matplotlib)},
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("Controller environment is incomplete. Activate go2_isaac_controller.\n"
                           + result.stderr[-6000:])
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError("Controller dependency check returned invalid output:\n" + result.stdout[-3000:]) from error


def runtime_command(runtime, arguments):
    if os.name == "nt" and runtime.suffix.lower() in {".bat", ".cmd"}:
        # cmd expands metacharacters even when shell=False launches a batch file.
        # Reject unusual paths instead of permitting argument interpretation.
        values = [str(runtime), *map(str, arguments)]
        if any(any(char in arg for char in '&|<>^%!"\r\n') for arg in values):
            raise ValueError("Windows batch paths must not contain shell metacharacters (& | < > ^ % ! or quotes). "
                             "Place the repository at a path without those characters.")
        command = '"' + " ".join('"' + arg + '"' for arg in values) + '"'
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        # Pass the cmd command line directly: list2cmdline's C-runtime quote
        # escaping is not the grammar used by cmd.exe for its /c argument.
        return '"' + comspec + '" /d /s /c ' + command
    return [str(runtime), *map(str, arguments)]


def run_logged(command, log_path, env):
    """Stream progress to the prompt and a durable per-process diagnostic log."""
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("Command: " + json.dumps(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def inspect_case_execution(output, case, run_id, duration_s, *, require_preflight=False, process_exit_code=None):
    """Check durable evidence even when Kit incorrectly returns OS exit code 0.

    This is an execution/startup gate, NOT a replacement for the numerical
    report. A measured fall is a valid execution, including negative controls.
    Missing, failed or stale preflight blocks subsequent cases.
    """
    from isaac_validation.evidence import inspect_native_log, log_errors, exit_code_text
    output = Path(output)
    errors = log_errors(inspect_native_log(output/'logs'/f'{case}.log'))
    if process_exit_code is not None and (type(process_exit_code) is not int or process_exit_code != 0):
        errors.append('Observed process exit: ' + exit_code_text(process_exit_code))

    def read_record(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            return {}

    record = read_record(output / case / "case.json")
    if record:
        if record.get("run_id") != run_id or record.get("case", record.get("name")) != case:
            errors.append("case.json does not match this case/run ID")
        if record.get("engine") != "Isaac Sim / PhysX":
            errors.append("case.json does not identify an Isaac Sim / PhysX execution")
        if type(record.get("execution_exit_code")) is not int or record["execution_exit_code"] != 0:
            errors.append(f"Recorded pre-close execution_exit_code={record.get('execution_exit_code')!r}")
        reason = record.get("termination_reason")
        if reason not in {"duration", "fall"}:
            errors.append(f"Case did not terminate physically: {record.get('failure') or reason}")
        elif reason == "duration" and (record.get("completed") is not True or record.get("failure")):
            errors.append("Duration termination was not a completed, error-free rollout")
        elapsed = record.get("simulated_s")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or not 0 < elapsed <= duration_s + .0011):
            errors.append("No valid positive simulated duration was recorded")
        elif reason == "duration" and abs(elapsed - duration_s) > .0011:
            errors.append("Recorded simulation did not cover the requested duration")
        if record.get("duration_s") != duration_s:
            errors.append("case.json duration does not match the requested experiment")
        if not record.get("source_sha256"):
            errors.append("case.json has no source fingerprints")
    else:
        errors.append("No usable case execution record")

    trace = output / case / "trajectory.npz"
    if not trace.is_file() or trace.stat().st_size == 0:
        errors.append("No nonempty trajectory.npz was produced")
    if require_preflight:
        for name in ("mechanics.json", "reference_validation.json"):
            certificate = read_record(output / name)
            if certificate.get("passed") is not True or certificate.get("run_id") != run_id:
                errors.append(f"{name}: current-run preflight did not pass")
            if (not record.get("source_sha256")
                    or certificate.get("source_sha256") != record.get("source_sha256")):
                errors.append(f"{name}: preflight source fingerprints do not match the case")
    return errors


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--isaac-root", type=Path, help="Installed binary distribution containing python.bat/python.sh")
    runtime.add_argument("--isaac-python", type=Path, help="Python executable of a separate Isaac Sim pip environment")
    parser.add_argument("--suite", choices=("smoke", "nominal", "full"), default="full")
    parser.add_argument("--headless", action="store_true", help="Render video without opening the Isaac GUI")
    parser.add_argument("--output", type=Path, help="New or empty output directory; defaults to a unique runs/ folder")
    parser.add_argument("--no-video", action="store_true", help="Diagnostic run without video; cannot satisfy a video acceptance gate")
    parser.add_argument("--ffmpeg", type=Path, help="Override the bundled imageio-ffmpeg encoder executable")
    parser.add_argument("--capture-mode", choices=("render", "replicator"), default="render",
                        help="Default render drains the blocking World renderer without Replicator frame scheduling")
    parser.add_argument("--shutdown-mode", choices=("auto", "fast", "graceful"), default="auto",
                        help="Auto uses fast shutdown only when the runtime preserves explicit exit codes")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runtime = ((args.isaac_root.expanduser().resolve() / ("python.bat" if os.name == "nt" else "python.sh"))
               if args.isaac_root else args.isaac_python.expanduser().absolute())
    if not runtime.is_file():
        print(f"Isaac runtime launcher does not exist: {runtime}", file=sys.stderr)
        return 2
    run_id = str(uuid.uuid4())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output.expanduser().resolve() if args.output else ROOT / "runs" / f"{stamp}_{run_id[:8]}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f"Output must be a new or empty directory: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    cases = list(FULL_CASES if args.suite == "full" else ("nominal",))
    manifest = dict(schema_version=2, run_id=run_id, suite=args.suite, started_utc=utc_now(),
                    output=str(output), package=str(ROOT), runtime=str(runtime),
                    controller_python=os.path.abspath(sys.executable), platform=platform.platform(),
                    duration_s=2.0 if args.suite == "smoke" else 26.0,
                    requested_cases=cases, headless=args.headless, video_requested=not args.no_video,
                    cases=[], status="running", capture_mode=args.capture_mode, shutdown_mode=args.shutdown_mode)
    manifest_path = output / "suite_manifest.json"
    write_json(manifest_path, manifest)
    failed = False
    interrupted = False
    try:
        probe = controller_probe()
        manifest["controller_environment"] = probe
        ffmpeg = args.ffmpeg.expanduser().resolve() if args.ffmpeg else Path(probe["ffmpeg"]).resolve()
        if not args.no_video and not ffmpeg.is_file():
            raise FileNotFoundError(f"FFmpeg executable is unavailable: {ffmpeg}")
        manifest["ffmpeg"] = str(ffmpeg)
        env = isaac_environment(os.environ, runtime if args.isaac_python else None)
        print(f"Run ID: {run_id}\nResults: {output}\nController Python: {sys.executable}", flush=True)
        for number, case in enumerate(cases):
            entry = dict(case=case, started_utc=utc_now(), status="running", log=f"logs/{case}.log")
            manifest["cases"].append(entry)
            write_json(manifest_path, manifest)
            cli = [str(ROOT / "run_isaac.py"), "--controller-python", os.path.abspath(sys.executable),
                   "--output", str(output), "--case", case, "--run-id", run_id,
                   "--duration", str(manifest["duration_s"]), "--capture-mode", args.capture_mode,
                   "--shutdown-mode", args.shutdown_mode]
            if number == 0:
                cli.append("--preflight")
            if args.headless:
                cli.append("--headless")
            if args.no_video:
                cli.append("--no-video")
            else:
                cli += ["--ffmpeg", str(ffmpeg)]
            print(f"\n[{number + 1}/{len(cases)}] Starting {case}", flush=True)
            code = run_logged(runtime_command(runtime, cli), logs / f"{case}.log", env)
            execution_errors = inspect_case_execution(
                output, case, run_id, manifest["duration_s"], require_preflight=(number == 0), process_exit_code=code)
            effective_code = code if code != 0 else (1 if execution_errors else 0)
            entry.update(process_exit_code=code, exit_code=effective_code,
                         execution_errors=execution_errors, finished_utc=utc_now(),
                         status="completed" if effective_code == 0 else "failed")
            for problem in execution_errors:
                print(f"{case}: {problem}", file=sys.stderr, flush=True)
            failed = failed or effective_code != 0
            write_json(manifest_path, manifest)
            # Kit fast shutdown can return 0 even after a Python exception.
            # Use both the OS status and the saved current-run evidence.
            if number == 0 and effective_code != 0:
                for blocked in cases[1:]:
                    manifest["cases"].append(dict(case=blocked, status="blocked",
                        reason="Initial execution or current-run preflight failed"))
                write_json(manifest_path, manifest)
                break
    except KeyboardInterrupt:
        interrupted = True
        failed = True
        manifest["error"] = "Interrupted by user"
    except Exception as error:
        failed = True
        manifest["error"] = f"{type(error).__name__}: {error}"
        print(manifest["error"], file=sys.stderr)
    finally:
        manifest["finished_utc"] = utc_now()
        manifest["status"] = "interrupted" if interrupted else ("failed" if failed else "executed")
        write_json(manifest_path, manifest)
    # Reporting evaluates acceptance separately from whether subprocesses exited.
    report_command = [sys.executable, "-m", "isaac_validation.report", "--output", str(output),
                      "--suite", args.suite, "--run-id", run_id]
    try:
        report_code = run_logged(report_command, logs / "report.log", os.environ.copy())
        manifest["report_exit_code"] = report_code
        failed = failed or report_code != 0
        report_path = output / "validation.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest["validation_status"] = report.get("status")
            manifest["full_validation"] = report.get("full_validation") is True
            manifest["smoke_passed"] = report.get("smoke_passed") is True
    except Exception as error:
        failed = True
        manifest["report_error"] = f"{type(error).__name__}: {error}"
        print(manifest["report_error"], file=sys.stderr)
    manifest["status"] = "interrupted" if interrupted else ("failed" if failed else "completed")
    write_json(manifest_path, manifest)
    print(f"\nResults and diagnostics: {output}", flush=True)
    return 130 if interrupted else (1 if failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
