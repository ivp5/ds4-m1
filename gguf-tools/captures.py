"""Validated streaming reader for the runtime's FFN-input capture format."""
import struct
from pathlib import Path
import numpy as np
E,NE,NL=5120,384,40
def records(path):
    with Path(path).open('rb') as f:
        while True:
            h=f.read(24)
            if not h:return
            if len(h)!=24:raise ValueError('truncated XCAP header')
            magic,step,L,rows,nt,ns=struct.unpack('<6I',h)
            if magic!=0x50414358 or not 0<=L<NL or not 1<=rows<=4 or (nt,ns)!=(NE,6):raise ValueError('unexpected XCAP dimensions')
            n=rows*(E*2+nt*4+ns*4);b=f.read(n)
            if len(b)!=n:raise ValueError('truncated XCAP record: do not train on unfinished captures')
            x=np.frombuffer(b,'<f2',rows*E).reshape(rows,E).copy()
            ids=np.frombuffer(b,'<i4',rows*ns,rows*(E*2+nt*4)).reshape(rows,ns).copy()
            if not np.isfinite(x).all() or (ids<0).any() or (ids>=NE).any():raise ValueError('invalid XCAP values')
            yield step,L,x,ids
