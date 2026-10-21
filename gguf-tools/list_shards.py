#!/usr/bin/env python3
"""List only the source shards a builder consumes; no network or data copying."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--component',choices=['bs64','draft'],required=True);a=p.parse_args()
    m=json.loads(a.index.read_text())['weight_map']
    def wanted(k):
        if a.component=='draft':return k.startswith('mtp.')
        return k.startswith('layers.') and '.ffn.experts.' in k and any(k.endswith('.'+w+'.'+s) for w in ('w1','w3') for s in ('weight','scale'))
    found=sorted({v for k,v in m.items() if wanted(k)})
    if not found:p.error('no tensors for requested component')
    print('\n'.join(found))
if __name__=='__main__':main()
