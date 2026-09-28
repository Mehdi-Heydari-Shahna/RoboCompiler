"""Run/resume a list of ovrun jobs with bounded concurrency (development only)."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get('KANGAROO_OVPHYSX_RUNS', 'runs'))


def main(job_file, parallel=2):
    jobs = json.loads(Path(job_file).read_text())
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    pending = [j for j in jobs if not (RUNS / j['name'] / 'result.json').exists()]
    running = []
    while pending or running:
        while pending and len(running) < parallel:
            j = pending.pop(0)
            cmd = [sys.executable, str(HERE / 'ovrun.py'), '--out', str(RUNS / j['name'])] + j['args']
            log = (RUNS / (j['name'] + '.log')).open('a')
            log.write('\n=== launch ' + time.strftime('%H:%M:%S') + ' ' + ' '.join(cmd) + '\n'); log.flush()
            running.append((j, subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        time.sleep(5)
        still = []
        for j, proc, log in running:
            if proc.poll() is None:
                still.append((j, proc, log))
            else:
                log.write(f'=== exit {proc.returncode}\n'); log.close()
        running = still


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
