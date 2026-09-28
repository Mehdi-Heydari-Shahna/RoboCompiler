"""Launch a benchmark workflow with the currently selected Python environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WORKFLOWS = {
    "mujoco": (".", "run_go2.py"),
    "generated": ("workflows/generated", "run_go2.py"),
    "pinocchio": ("workflows/pinocchio", "run_pinocchio.py"),
    "isaac": ("workflows/isaac", "launch.py"),
}


def workflow_command(name, arguments=(), python=None, root=ROOT):
    """Return the process directory and argument vector without invoking a shell."""
    directory, entry = WORKFLOWS[name]
    cwd = Path(root).resolve() / directory
    arguments = list(arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    return cwd, [python or sys.executable, str(cwd / entry), *arguments]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List workflow entry points.")
    parser.add_argument("--print-command", action="store_true", help="Print the command without executing it.")
    parser.add_argument("workflow", nargs="?", choices=WORKFLOWS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list:
        for name, (directory, script) in WORKFLOWS.items():
            print(f"{name:10s} {Path(directory) / script}")
        return 0
    if args.workflow is None:
        parser.error("select a workflow or use --list")
    cwd, command = workflow_command(args.workflow, args.arguments)
    if not Path(command[1]).is_file():
        parser.error(f"missing workflow entry point: {command[1]}")
    if args.print_command:
        print(json.dumps({"cwd": str(cwd), "command": command}, indent=2))
        return 0
    return subprocess.run(command, cwd=cwd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
