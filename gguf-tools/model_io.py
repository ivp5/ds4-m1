"""Small bounded GGUF/safetensors readers; never map the whole checkpoint."""
import hashlib, json, math, os, struct
from pathlib import Path
class GGUF:
    def __init__(self,path):
        self.path=Path(path); self.bytes=self.path.stat().st_size
        self.tensors={}; self.metadata={}
        with self.path.open('rb') as f:
            def read(n):
                if n<0 or f.tell()+n>self.bytes: raise ValueError('truncated GGUF')
                b=f.read(n)
                if len(b)!=n: raise ValueError('short GGUF read')
                return b
            def unpack(fmt): return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
            def string():
                n=unpack('Q')
                if n>64*1024*1024: raise ValueError('oversized GGUF string')
                return read(n).decode('utf8')
            forms={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
            def value(t,keep=False):
                if t==8:return string()
                if t==9:
                    et,n=unpack('I'),unpack('Q')
                    if n>100000000: raise ValueError('oversized GGUF array')
                    if et in forms:
                        nb=n*struct.calcsize('<'+forms[et])
                        if f.tell()+nb>self.bytes: raise ValueError('truncated array')
                        f.seek(nb,1)
                    else:
                        for _ in range(n):value(et)
                    return None
                if t not in forms:raise ValueError('unknown GGUF metadata type')
                return unpack(forms[t])
            if read(4)!=b'GGUF' or unpack('I') not in (2,3):raise ValueError('expected GGUF v2/v3')
            nt,nkv=unpack('Q'),unpack('Q')
            if max(nt,nkv)>1000000:raise ValueError('oversized directory')
            for _ in range(nkv):
                name=string();v=value(unpack('I'))
                if name in ('general.alignment','general.architecture','general.name'):self.metadata[name]=v
            for _ in range(nt):
                name=string();nd=unpack('I')
                if nd<1 or nd>4:raise ValueError('invalid tensor rank')
                dims=tuple(unpack('Q') for _ in range(nd));ty,off=unpack('I'),unpack('Q')
                if name in self.tensors or min(dims)<=0:raise ValueError('duplicate/empty tensor')
                self.tensors[name]={'shape':dims,'type':ty,'offset':off}
            align=self.metadata.get('general.alignment',32)
            if align<=0 or align&(align-1):raise ValueError('invalid alignment')
            self.data=(f.tell()+align-1)//align*align
            for t in self.tensors.values():
                t['offset']+=self.data
                if t['offset']>=self.bytes:raise ValueError('tensor offset outside GGUF')
            f.seek(0);self.header_sha256=hashlib.sha256(read(self.data)).hexdigest()
    def identity(self):return {'bytes':self.bytes,'header_sha256':self.header_sha256}
    def tensor(self,name,shape=None,ty=None):
        t=self.tensors[name]
        if shape is not None and tuple(shape)!=t['shape']:raise ValueError(f'{name}: shape {t["shape"]}, expected {shape}')
        if ty is not None and ty!=t['type']:raise ValueError(f'{name}: GGML type {t["type"]}, expected {ty}; use the matching Q2 conversion')
        return t

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def receipt(out,kind,model,**extra):
    p=Path(out); data={'format':kind,'bytes':p.stat().st_size,'model':model.identity(),**extra}
    q=p.with_name(p.name+'.json');q.write_text(json.dumps(data,indent=2)+'\n');return data
class SafeTensors:
    def __init__(self,root):
        self.root=Path(root);self.mapping=json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map'];self.headers={}
    def raw(self,name):
        rel=self.mapping[name];p=(self.root/rel).resolve()
        if not p.is_relative_to(self.root.resolve()):raise ValueError('shard path escapes source directory')
        with p.open('rb') as f:
            if rel not in self.headers:
                b=f.read(8)
                if len(b)!=8:raise ValueError('short safetensors header')
                n=struct.unpack('<Q',b)[0]
                if n>100000000:raise ValueError('oversized safetensors header')
                self.headers[rel]=(8+n,json.loads(f.read(n)))
            base,h=self.headers[rel];t=h[name];lo,hi=t['data_offsets']
            if not 0<=lo<=hi<=p.stat().st_size-base:raise ValueError('invalid tensor span')
            f.seek(base+lo);b=f.read(hi-lo)
            if len(b)!=hi-lo:raise ValueError('short tensor read')
            return b,t
