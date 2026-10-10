"""Windows/Linux pipe client for the isolated Miniforge controller service.

This module deliberately uses only Python's standard library. Import it in
Isaac's Python without importing Pinocchio, SciPy, OSQP, or the PACDM module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading


class WorkerError(RuntimeError):
    """The controller worker failed or could not reply within its timeout."""


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class WorkerClient:
    def __init__(self, python_executable, root, log_path, timeout=60.0):
        self.root = Path(root).expanduser().resolve()
        self.log_path = Path(log_path).expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve a venv's symlink path: resolving it to the base interpreter
        # silently discards that environment's site-packages on Linux.
        executable = Path(os.path.abspath(Path(python_executable).expanduser()))
        if not executable.is_file():
            raise FileNotFoundError(f"Controller Python does not exist: {executable}")
        if not (self.root / "isaac_validation/worker.py").is_file():
            raise FileNotFoundError(f"Controller worker package does not exist under {self.root}")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("Worker timeout must be positive")
        self._counter = 0
        self._closed = False
        self._lock = threading.Lock()
        self._replies = queue.Queue()
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "LD_LIBRARY_PATH",
                    "LD_PRELOAD", "DYLD_LIBRARY_PATH"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Prevent nested BLAS thread pools from dominating each small QP.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            env[key] = "1"
        search = [str(executable.parent)]
        if os.name == "nt":
            search.extend(str(executable.parent / name) for name in ("Scripts", "Library/bin"))
        env["PATH"] = os.pathsep.join(search + [env.get("PATH", "")])
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        try:
            self.process = subprocess.Popen(
                [str(executable), "-u", "-m", "isaac_validation.worker"],
                cwd=str(self.root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception:
            self._log.close()
            raise
        self._reader = threading.Thread(target=self._read_stdout, name="PACDM-worker-reader", daemon=True)
        self._reader.start()

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self._replies.put(line)
        except Exception as error:
            self._replies.put(error)
        finally:
            self._replies.put(None)

    def _diagnostic(self, message):
        try:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        except OSError:
            tail = "(worker diagnostic log is unavailable)"
        return WorkerError(f"{message}\nController log: {self.log_path}\n{tail}")

    def request(self, op, timeout=None, **parameters):
        with self._lock:
            if self._closed:
                raise WorkerError("Controller worker is closed")
            if self.process.poll() is not None:
                raise self._diagnostic(f"Controller worker exited with code {self.process.returncode}")
            self._counter += 1
            request_id = self._counter
            encoded = json.dumps(dict(id=request_id, op=op, **parameters),
                                 default=_json_default, allow_nan=False)
            try:
                self.process.stdin.write(encoded + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise self._diagnostic(f"Cannot send controller request {op}: {error}") from error
            try:
                line = self._replies.get(timeout=self.timeout if timeout is None else float(timeout))
            except queue.Empty as error:
                # A late reply must never be mistaken for the next timestep.
                self._abort()
                raise self._diagnostic(f"Controller request {op!r} timed out") from error
            if line is None or isinstance(line, Exception):
                self._abort()
                raise self._diagnostic(f"Controller pipe closed during request {op!r}: {line}")
            try:
                response = json.loads(line)
            except (ValueError, TypeError) as error:
                self._abort()
                raise self._diagnostic(f"Invalid JSON from controller: {line[:500]!r}") from error
            if not isinstance(response, dict) or response.get("id") != request_id:
                self._abort()
                raise self._diagnostic(f"Controller reply ID mismatch for {request_id}: {response!r}")
            if not response.get("ok"):
                raise self._diagnostic(
                    f"Controller request {op!r} failed: {response.get('error')}\n{response.get('traceback', '')}")
            return response["result"]

    def init(self, duration=26.0, feedforward=True, **parameters):
        return self.request("init", duration=duration, feedforward=feedforward, **parameters)

    def command(self, t, q, v):
        return self.request("command", t=float(t), q=q, v=v)

    def audit_state(self, q, v):
        return self.request("audit_state", q=q, v=v)

    def validate_reference(self):
        return self.request("validate_reference", timeout=600.0)

    def _abort(self):
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)

    def close(self):
        if not self._closed:
            try:
                self.request("shutdown", timeout=2.0)
            except Exception:
                pass
            self._abort()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
