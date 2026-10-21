#!/usr/bin/env python3
"""Reconstruct checked Metal variants from small patches against upstream-layout sources."""
import argparse,hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
from graphlib import TopologicalSorter
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(root=ROOT,check=False):
    source=root/'metal/patches';out=root/'build/metal'
    manifest=json.loads((source/'manifest.json').read_text())
    if not check:out.mkdir(parents=True,exist_ok=True)
    patcher=shutil.which('patch')
    if not patcher and not check:raise ValueError('install the standard patch utility to build Metal variants')
    dependencies={}
    for name,e in manifest.items():
        parent=Path(e['base']).name if e['base'].startswith('build/metal/') else None
        if parent and (parent not in manifest or e['base']!='build/metal/'+parent):
            raise ValueError('unknown generated kernel base: '+e['base'])
        dependencies[name]=[parent] if parent else []
    order=list(TopologicalSorter(dependencies).static_order())
    made=0
    for name in order:
        e=manifest[name]
        if Path(name).name!=name or Path(e['patch']).name!=e['patch'] or (not e['base'].startswith(('metal/','build/metal/')) or '..' in Path(e['base']).parts):
            raise ValueError('invalid kernel manifest path')
        base=root/e['base'];patch=source/e['patch'];dest=out/name
        if digest(base)!=e['base_sha256']:raise ValueError(f"{e['base']}: base changed; rebase the kernel patch explicitly")
        if digest(patch)!=e['patch_sha256']:raise ValueError(f"{e['patch']}: patch checksum mismatch")
        if dest.exists() and digest(dest)==e['output_sha256']:continue
        if check:raise ValueError(f'{name}: missing or stale; run make metal-variants')
        headers=patch.read_text().splitlines()[:2]
        if headers!=['--- a/'+name,'+++ b/'+name]:raise ValueError('unexpected patch target')
        with tempfile.TemporaryDirectory(prefix='kernel-',dir=out.parent) as td:
            td=Path(td);target=td/name;shutil.copyfile(base,target)
            env=dict(os.environ,TMPDIR=str(td))
            p=subprocess.run([patcher,'-s','-f','-F','0','-p1','-d',str(td),'-i',str(patch.resolve())],
                             env=env,capture_output=True,text=True,timeout=15)
            if p.returncode:raise ValueError(f'{name}: patch failed: {p.stderr.strip()} {p.stdout.strip()}')
            if target.stat().st_size!=e['output_bytes'] or digest(target)!=e['output_sha256']:
                raise ValueError(f'{name}: generated kernel differs from the validated source')
            os.replace(target,dest);made+=1
    return made,len(manifest)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    try:made,total=build(check=a.check)
    except (OSError,ValueError,subprocess.TimeoutExpired) as e:p.error(str(e))
    print(f'Metal variants: {total} verified, {made} generated')
if __name__=='__main__':main()
