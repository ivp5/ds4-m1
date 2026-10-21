"""BS64 reference codec and optional float64 Torch acceleration."""
import numpy as np
ROWS,COLS,G,SB,NE,NL=2304,5120,64,512,384,40
NG,NSB=COLS//G,COLS//SB
ROWB=COLS//4+NG+4*NSB
MATB=ROWS*ROWB;EXPB=2*MATB;LAYB=NE*EXPB;HDR=16384
def encode_t(W32, imp):
    W = W32.to(T64); A = W.reshape(ROWS, NG, G); P = imp.reshape(1, NG, G)
    lo0, hi0 = A.amin(dim=2, keepdim=True), A.amax(dim=2, keepdim=True)
    best = bs = bl = None
    for f in (0.7, 0.8, 0.9, 1.0):
        for sh in (0.0, 0.1, -0.1):
            s = torch.clamp((hi0 - lo0) / 3 * f, min=1e-12); lo = lo0 + sh * s
            q = torch.clamp(torch.round((A - lo) / s), 0, 3); e = (P * (lo + s * q - A) ** 2).sum(dim=2, keepdim=True)
            if best is None: best, bs, bl = e, s, lo
            else: m = e < best; best, bs, bl = torch.where(m, e, best), torch.where(m, s, bs), torch.where(m, lo, bl)
    per = SB // G; S = bs.reshape(-1, per); Mn = torch.clamp(-bl, min=0).reshape(-1, per)
    ds16 = (torch.clamp(S.amax(dim=1, keepdim=True), min=1e-12) / 15).to(torch.float16); dm16 = (torch.clamp(Mn.amax(dim=1, keepdim=True), min=1e-12) / 15).to(torch.float16)
    dsf, dmf = ds16.to(T64), dm16.to(T64)
    cs = torch.clamp(torch.round(S / torch.clamp(dsf, min=1e-30)), 1, 15); cm = torch.clamp(torch.round(Mn / torch.clamp(dmf, min=1e-30)), 0, 15)
    Sq = (cs * dsf).reshape(-1, 1); Mq = (cm * dmf).reshape(-1, 1)
    A2 = A.reshape(-1, G); qf = torch.clamp(torch.round((A2 + Mq) / torch.where(Sq > 0, Sq, 1.0)), 0, 3); nbad = int((~torch.isfinite(qf)).sum()); q = torch.nan_to_num(qf, nan=0.0).long()
    Wq = (Sq * q.to(T64) - Mq).reshape(W.shape)
    d = (Wq - W) ** 2; iw = imp.reshape(1, COLS); ew = float(torch.sqrt((d * iw).sum() / ((W ** 2) * iw).sum())); ep = float(torch.sqrt(d.sum() / (W ** 2).sum()))
    q32 = q.reshape(-1, 32); bit = torch.arange(32, device=dev, dtype=torch.long)[None, :]
    p0 = ((q32 & 1) << bit).sum(dim=1); p1 = ((q32 >> 1) << bit).sum(dim=1)
    planes = torch.stack([p0, p1], dim=1).reshape(ROWS, -1).to(torch.int32)          # int64 -> int32 wraps: the uint32 bit pattern
    codes = (cs.to(torch.uint8).reshape(-1) | (cm.to(torch.uint8).reshape(-1) << 4)).reshape(ROWS, NG)
    D = torch.stack([ds16.reshape(-1), dm16.reshape(-1)], dim=1).reshape(ROWS, NSB * 2)
    rec = np.concatenate([planes.cpu().numpy().view(np.uint8).reshape(ROWS, -1), codes.cpu().numpy(), D.cpu().numpy().view(np.uint8).reshape(ROWS, -1)], axis=1)
    assert rec.shape == (ROWS, ROWB)
    return rec.tobytes(), ew, ep, nbad

def encode_np(W, imp):   # Canonical NumPy codec
    A = W.reshape(ROWS, NG, G).astype(np.float64); P = imp.reshape(1, NG, G)
    lo0, hi0 = A.min(axis=2, keepdims=True), A.max(axis=2, keepdims=True)
    best = bs = bl = None
    for f in (0.7, 0.8, 0.9, 1.0):
        for sh in (0.0, 0.1, -0.1):
            s = np.maximum((hi0 - lo0) / 3 * f, 1e-12); lo = lo0 + sh * s
            q = np.clip(np.round((A - lo) / s), 0, 3); e = (P * (lo + s * q - A) ** 2).sum(axis=2, keepdims=True)
            if best is None: best, bs, bl = e, s, lo
            else: m = e < best; best, bs, bl = np.where(m, e, best), np.where(m, s, bs), np.where(m, lo, bl)
    per = SB // G; S = bs.reshape(-1, per); Mn = np.maximum(-bl, 0).reshape(-1, per)
    ds16 = (np.maximum(S.max(axis=1, keepdims=True), 1e-12) / 15).astype(np.float16); dm16 = (np.maximum(Mn.max(axis=1, keepdims=True), 1e-12) / 15).astype(np.float16)
    dsf, dmf = ds16.astype(np.float64), dm16.astype(np.float64)
    cs = np.clip(np.round(S / np.maximum(dsf, 1e-30)), 1, 15); cm = np.clip(np.round(Mn / np.maximum(dmf, 1e-30)), 0, 15)
    Sq = (cs * dsf).reshape(-1, 1); Mq = (cm * dmf).reshape(-1, 1)
    A2 = A.reshape(-1, G); q = np.clip(np.round((A2 + Mq) / np.where(Sq > 0, Sq, 1.0)), 0, 3).astype(np.uint32)
    q32 = q.reshape(-1, 32); bit = np.arange(32, dtype=np.uint32)[None, :]
    p0 = ((q32 & 1) << bit).sum(axis=1, dtype=np.uint64).astype(np.uint32); p1 = ((q32 >> 1) << bit).sum(axis=1, dtype=np.uint64).astype(np.uint32)
    planes = np.stack([p0, p1], axis=1).reshape(ROWS, -1)
    codes = (cs.astype(np.uint8).ravel() | (cm.astype(np.uint8).ravel() << 4)).reshape(ROWS, NG)
    D = np.stack([ds16.ravel(), dm16.ravel()], axis=1).reshape(ROWS, NSB * 2)
    rec = np.concatenate([planes.view(np.uint8).reshape(ROWS, -1), codes, D.view(np.uint8).reshape(ROWS, -1)], axis=1)
    return rec.tobytes()
