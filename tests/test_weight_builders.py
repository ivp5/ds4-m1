"""Small CPU tests; no model, GPU, downloads, or files outside build/checks."""
import importlib.util,json,struct,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'gguf-tools'));sys.path.insert(0,str(ROOT/'metal'))
from model_io import GGUF
from build_dense import pack,quant_eng
from dense_codec import quant
from build_bank import choose
from build_lookahead import fit
from trace_rates import rates
from build_kernels import build
from update_kernel import entry
import bs64_codec as bs
class Tools(unittest.TestCase):
    def test_kernel_build_and_drift_guards(self):
        local=ROOT/'build/checks';local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as d:
            r=Path(d);(r/'metal').mkdir();(r/'metal/patches').mkdir(parents=True)
            base=b'line one\nline two\n';want=b'line one\nline changed\n'
            (r/'metal/toy.metal').write_bytes(base)
            patch,e=entry('toy_variant.metal','metal/toy.metal',base,want)
            pd=r/'metal/patches';(pd/e['patch']).write_text(patch);(pd/'manifest.json').write_text(json.dumps({'toy_variant.metal':e}))
            self.assertEqual(build(r),(1,1));self.assertEqual(build(r,check=True),(0,1))
            generated=r/'build/metal/toy_variant.metal';self.assertEqual(generated.read_bytes(),want)
            generated.write_bytes(b'broken')
            with self.assertRaises(ValueError):build(r,check=True)
            self.assertEqual(build(r),(1,1))
            (r/'metal/toy.metal').write_bytes(base+b'changed base\n')
            with self.assertRaises(ValueError):build(r)
            (r/'metal/toy.metal').write_bytes(base);(pd/e['patch']).write_text(patch+'corrupt\n')
            with self.assertRaises(ValueError):build(r)
    def test_kernel_dependencies(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/checks') as d:
            r=Path(d);(r/'metal/patches').mkdir(parents=True)
            base=b'base\n';a=b'base\nshared\n';b=a+b'child\n'
            (r/'metal/toy.metal').write_bytes(base)
            manifest={}
            for name,parent,source,want in [('a.metal','metal/toy.metal',base,a),('b.metal','build/metal/a.metal',a,b)]:
                patch,e=entry(name,parent,source,want);manifest[name]=e
                (r/'metal/patches'/e['patch']).write_text(patch)
            mp=r/'metal/patches/manifest.json'
            # The child deliberately precedes its parent in the stored mapping.
            mp.write_text(json.dumps(dict(reversed(list(manifest.items())))))
            self.assertEqual(build(r),(2,2));self.assertEqual((r/'build/metal/b.metal').read_bytes(),b)
            manifest['a.metal']['base']='build/metal/b.metal';mp.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):build(r)
            manifest['a.metal']['base']='build/metal/missing.metal';mp.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):build(r)
    def test_trace_transitions(self):
        lines=['2026-01-01 00:00:01.000 prefill sync done tool_round=0 prompt=10 cached=0 suffix=10 rc=0 500 ms',
               '2026-01-01 00:00:03.000 generation finished tool_round=0 generated=20']
        a=rates(lines);self.assertEqual(a['prefill_rows_per_second'],20);self.assertEqual(a['generation_tokens_per_second'],10)
        self.assertEqual(a['tokens_per_prefill_plus_generation_second'],8)
        for bad in [lines[:1],lines+lines,[lines[0],lines[1].replace('round=0','round=1')],[lines[0].replace('rc=0','rc=1'),lines[1]]]:
            with self.assertRaises(ValueError):rates(bad)
    def test_gguf_bounds_and_identity(self):
        def string(s):b=s.encode();return struct.pack('<Q',len(b))+b
        hdr=b'GGUF'+struct.pack('<IQQ',3,1,0)+string('tensor')+struct.pack('<IQQIQ',2,32,2,8,0)
        hdr+=b'\0'*((-len(hdr))%32)
        local=ROOT/'build/checks';local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as d:
            f=Path(d)/'toy.gguf';f.write_bytes(hdr+bytes(68));g=GGUF(f)
            self.assertEqual(g.tensor('tensor',(32,2),8)['offset'],len(hdr))
            self.assertEqual(len(g.header_sha256),64)
            with self.assertRaises(ValueError):g.tensor('tensor',ty=10)
            f.write_bytes(b'GGUF');self.assertRaises(ValueError,GGUF,f)
    def test_q4l_roundtrip(self):
        x=np.random.default_rng(55).normal(size=(16,32)).astype(np.float32)
        d,q=quant(x,np.ones_like(x));b=pack(q,d,256,2)
        n=b[:,:128];back=np.stack([n&15,n>>4],axis=2).reshape(16,32).astype(np.int16)-8
        np.testing.assert_array_equal(back,q)
        np.testing.assert_array_equal(b[:,128:].copy().view('<f2').reshape(-1),d)
        z=np.zeros((8,32),np.float32);d,q=quant(z);self.assertTrue((d==0).all() and (q==0).all())
    def test_engram_zero(self):
        d,q=quant_eng(np.zeros((8,32),np.float32));self.assertTrue(np.isfinite(d).all());self.assertTrue((q==0).all())
    def test_bank_floor_and_ties(self):
        k=choose([[0]*128 for _ in range(3)])
        self.assertEqual(k[0],set(range(128)));self.assertEqual(k[1],{0,1,2});self.assertEqual(k[2],{0,1,2})
        k=choose([[i+s for i in range(128)] for s in range(3)])
        self.assertEqual(sum(map(len,k)),128)
    def test_bs64_small_and_zero(self):
        old={k:getattr(bs,k) for k in ['ROWS','COLS','NG','NSB','ROWB']}
        try:
            bs.ROWS=2;bs.COLS=512;bs.NG=8;bs.NSB=1;bs.ROWB=140
            for w in [np.random.default_rng(2).normal(size=(2,512)).astype(np.float32),np.zeros((2,512),np.float32)]:
                b=bs.encode_np(w,np.ones(512));self.assertEqual(len(b),280)
                self.assertEqual(b,bs.encode_np(w,np.ones(512)))
        finally:
            for k,v in old.items():setattr(bs,k,v)
    def test_ridge_identity(self):
        r=np.random.default_rng(9);x=r.normal(size=(50,8));w=r.normal(size=(8,4))
        np.testing.assert_allclose(fit(x,x,w,300),w,atol=0,rtol=0)
        xt=x+.01*r.normal(size=x.shape);a=fit(x,xt,w,300)
        lhs=(x.T@x+300*np.eye(8))@(a-w);rhs=x.T@(xt@w-x@w)
        np.testing.assert_allclose(lhs,rhs,atol=1e-11)
if __name__=='__main__':unittest.main()
