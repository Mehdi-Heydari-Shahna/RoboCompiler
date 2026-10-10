"""Consistent argument quoting for Windows Miniforge and Isaac Python launchers."""
import os
from pathlib import Path
import subprocess


def python_command(executable, script, arguments=()):
    executable = Path(executable).resolve()
    command = [str(executable), '-u', str(Path(script).resolve()), *map(str, arguments)]
    if os.name == 'nt' and executable.suffix.lower() in ('.bat', '.cmd'):
        if any(any(c in arg for c in '%!^&|<>\r\n"') for arg in command):
            raise ValueError('Use paths without cmd metacharacters, or pass an Isaac python.exe')
        prefix = subprocess.list2cmdline([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/s', '/c'])
        return prefix + ' "' + subprocess.list2cmdline(command) + '"'
    return command
