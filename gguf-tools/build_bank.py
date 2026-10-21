#!/usr/bin/env python3
"""Choose a resident drafter bank from calibration-only per-stage counts."""
import argparse,json
from pathlib import Path
from model_io import sha256

def choose(counts,total=128,minimum=3):
    if not 1<=total<=384:raise ValueError('bank size out of range')
    order=sorted(((s,i) for s in range(3) for i in range(128)),key=lambda k:(-counts[k[0]][k[1]],*k))
    keep=[set() for _ in range(3)]
    for s,i in order[:total]:keep[s].add(i)
    for s in range(3):
        for stage,i in order[total:]:
            if len(keep[s])>=minimum:break
            if stage==s:keep[s].add(i)
    return keep

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--counts',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    counts=[[0]*128 for _ in range(3)]
    for file in a.counts:
        rows=[[int(x) for x in l.split()] for l in file.read_text().splitlines()]
        if len(rows)!=3 or any(len(r)!=128 or min(r)<0 for r in rows):p.error('counts must be three rows of 128 nonnegative integers')
        for s in range(3):counts[s]=[x+y for x,y in zip(counts[s],rows[s])]
    if not sum(map(sum,counts)):p.error('no router selections captured')
    if a.out.exists():p.error('output exists')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(''.join(' '.join(map(str,sorted(k)))+'\n' for k in choose(counts)))
    a.out.with_name(a.out.name+'.json').write_text(json.dumps({'counts_sha256':[sha256(x) for x in a.counts]},indent=2)+'\n')
if __name__=='__main__':main()
