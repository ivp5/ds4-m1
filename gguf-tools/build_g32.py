#!/usr/bin/env python3
"""Build the column-major G32 routed-down overlay from a compatible Q2_K GGUF."""
import argparse, os, subprocess, shutil
from pathlib import Path
from model_io import GGUF, receipt
BYTES=16384+40*384*3502080

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--plan',action='store_true');a=p.parse_args()
    model=GGUF(a.model);offs=[model.tensor(f'blk.{i}.ffn_down_exps.weight',(2304,5120,384),10)['offset'] for i in range(40)]
    print(f'Output: {BYTES:,} bytes. Requantization changes weights; this is not lossless.',flush=True)
    if a.plan:return
    a.out=a.out.resolve();a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():p.error('output exists; choose a new output (never overwrite model assets)')
    if shutil.disk_usage(a.out.parent).free<BYTES+(1<<30):p.error('insufficient free space')
    build=Path(__file__).resolve().parents[1]/'build';build.mkdir(exist_ok=True)
    tmp=build/'tmp';tmp.mkdir(exist_ok=True);env=dict(os.environ,TMPDIR=str(tmp))
    offset_file=a.out.with_name(a.out.name+'.offsets.txt');offset_file.write_text('\n'.join(map(str,offs))+'\n')
    for name in ('make_g32_down','verify_g32_down'):
        subprocess.run([os.environ.get('CC','clang'),'-O3','-o',str(build/name),str(Path(__file__).with_name(name+'.c')),'-lpthread','-lm'],env=env,check=True)
    part=a.out.with_name(a.out.name+'.part')
    if part.exists():p.error('partial output exists; inspect/retire it before retrying')
    subprocess.run([str(build/'make_g32_down'),str(a.model.resolve()),str(offset_file),str(part),str(a.threads)],check=True)
    if part.stat().st_size!=BYTES:raise ValueError('wrong G32 file size')
    subprocess.run([str(build/'verify_g32_down'),str(a.model.resolve()),str(offset_file),str(part)],check=True)
    part.rename(a.out);receipt(a.out,'G32D-v1',model,verification='sampled roundtrip and corruption canary; not a fidelity test')
if __name__=='__main__':main()
