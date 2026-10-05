"""Small read-only inspection helper for the independent annotation review."""
import json
from pathlib import Path
import sys


def main():
    path = Path(sys.argv[2])
    if sys.argv[1] == 'lines':
        lines = path.read_text(encoding='utf-8').splitlines()
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        stop = int(sys.argv[4]) if len(sys.argv) > 4 else len(lines)
        for i, line in enumerate(lines[start-1:stop], start):
            print(f'{i}: {line}')
    elif sys.argv[1] == 'shape':
        obj = json.loads(path.read_text(encoding='utf-8').splitlines()[0]) if path.suffix == '.jsonl' else json.loads(path.read_text(encoding='utf-8'))
        def shape(x, depth=0):
            if isinstance(x, dict): return {k:shape(v, depth+1) for k,v in x.items()}
            if isinstance(x, list): return {'length':len(x),'first':shape(x[0],depth+1) if x else None}
            return type(x).__name__
        print(json.dumps(shape(obj),indent=2))


if __name__ == '__main__': main()
