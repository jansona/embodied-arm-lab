"""Create a source-only experimental snapshot; never packages runtime/config."""
from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parent

def package(destination):
    destination=Path(destination)
    roots=('assets','static','static_so101','tests')
    files=[p for p in ROOT.iterdir() if p.is_file() and (p.suffix in ('.py','.md','.sh','.txt') or p.name=='.gitignore')]
    example=ROOT/'config.example.json'
    if example.exists():
        data=json.loads(example.read_text())
        if data.get('api_key') not in ('',None):raise ValueError('Example contains nonempty credential')
        files.append(example)
    for name in roots:
        directory=ROOT/name
        if directory.exists():files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink())
    if any(p.is_symlink() or not p.is_relative_to(ROOT) for p in files):raise ValueError('Unsafe package path')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):archive.write(path,Path('jev-arm-sandbox')/path.relative_to(ROOT))
    return len(set(files)),destination.stat().st_size

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args()
    count,size=package(args.destination);print(f'{count} source files; {size} bytes')
