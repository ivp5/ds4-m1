#!/usr/bin/env python3
"""Report distinct prefill, generation and combined rates from fresh agent traces."""
import argparse,json,re
from datetime import datetime
from pathlib import Path
PF=re.compile(r"prefill sync done tool_round=(\d+) prompt=\d+ cached=\d+ suffix=(\d+) rc=(-?\d+) ([\d.]+) ms")
GEN=re.compile(r"generation finished tool_round=(\d+) generated=(\d+)")
def rates(lines):
    pending=None;seen=set();pf=gt=0;ps=gs=0.0;rounds=0
    for line in lines:
        m=PF.search(line)
        if m:
            rid,n,rc,ms=m.groups();rid=int(rid)
            if int(rc):raise ValueError('failed prefill; session is not a successful sample')
            if pending is not None or rid in seen:raise ValueError('unfinished or repeated round; use a fresh trace per session')
            t=datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S.%f')
            pending=(rid,int(n),float(ms)/1000,t);continue
        m=GEN.search(line)
        if m:
            rid,n=map(int,m.groups())
            if pending is None or rid!=pending[0]:raise ValueError('generation without its matching prefill')
            elapsed=(datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S.%f')-pending[3]).total_seconds()
            if elapsed<=0 or pending[2]<0:raise ValueError('nonpositive elapsed generation time or negative prefill time')
            seen.add(rid);pf+=pending[1];ps+=pending[2];gt+=n;gs+=elapsed;rounds+=1;pending=None
    if pending is not None:raise ValueError('unfinished generation round')
    if not rounds or ps<=0 or gs<=0:raise ValueError('no complete measurable session')
    return dict(rounds=rounds,prefill_rows=pf,prefill_seconds=ps,generated_tokens=gt,generation_seconds=gs,
                prefill_rows_per_second=pf/ps,generation_tokens_per_second=gt/gs,
                tokens_per_prefill_plus_generation_second=gt/(ps+gs))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('traces',type=Path,nargs='+');a=p.parse_args()
    for f in a.traces:
        try:
            with f.open() as inp:row=rates(inp)
        except (OSError,ValueError) as e:p.error(f'{f.name}: {e}')
        print(json.dumps(dict(trace=str(f),**row)))
if __name__=='__main__':main()
