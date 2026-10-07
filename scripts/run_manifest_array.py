"""Translate a Slurm array slot into a verified manifest job."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    mapping = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    manifest = json.loads(Path(mapping['manifest']).read_text(encoding='utf-8'))
    if mapping['manifest_hash'] != manifest['content_hash']:
        raise ValueError('Array mapping refers to another manifest.')
    index = mapping['workers'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    os.environ['NEUROSYM_ATTEMPT_TOKEN'] = mapping['attempt']
    command = [sys.executable, '-I', '-B', '-u', str(Path(__file__).with_name('run_analysis.py')),
               'worker', mapping['manifest'], '--index', str(index), *sys.argv[2:]]
    # Preserve the batch PID so Slurm's B:USR1 signal reaches the safe-point handler.
    os.execv(sys.executable, command)


if __name__ == '__main__':
    sys.exit(main())
