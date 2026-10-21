"""Importance-weighted Q4L quantizer; float16 scales are stored, never refitted."""
import numpy as np
VAR="msei"
def quant(x, qw=None):   # x (m, 32) float32, qw (m, 32) importance or None -> (d f16 (m,), int8 levels (m, 32))
    m = len(x); ax = np.abs(x); ia = ax.argmax(1); mx = x[np.arange(m), ia]; amax = ax[np.arange(m), ia]
    ok = amax > 1e-30; mxs = np.where(ok, mx, 1.0).astype(np.float32)
    w = (x * x) if VAR == "nlw" else np.ones_like(x)
    if qw is not None: w = qw * np.sqrt((x * x).mean(1, keepdims=True) + x * x)   # llama.cpp's imatrix weight
    best = np.full(m, -1.0, dtype=np.float32); scale = np.zeros(m, dtype=np.float32)
    if VAR in ("mse", "msei"):
        for i in range(-9, 10):
            isc = -(8.0 + 0.1 * i) / mxs
            L = np.clip(np.rint(x * isc[:, None]), -8, 7)
            sxl = (w * x * L).sum(1); sl2 = (w * L * L).sum(1)
            val = np.where(sl2 > 0, sxl * sxl / np.where(sl2 > 0, sl2, 1.0), -1.0)
            upd = val > best; best = np.where(upd, val, best); scale = np.where(upd, sxl / np.where(sl2 > 0, sl2, 1.0), scale)
        d = scale.astype(np.float16); df = d.astype(np.float32)
        L = np.clip(np.rint(x / np.where(df != 0, df, 1.0)[:, None]), -8, 7)
        return np.where(ok, d, np.float16(0)), np.where(ok[:, None], L, 0).astype(np.int8)
    for i in range(-7, 8):   # nl / nlw
        idd = (i + NL[0]) / mxs   # 1 / d
        Li = np.searchsorted(MID, x * idd[:, None]); q = NL[Li]
        sqx = (w * q * x).sum(1); sq2 = (w * q * q).sum(1)
        val = np.where(sq2 > 0, sqx * sqx / np.where(sq2 > 0, sq2, 1.0), -1.0)
        upd = val > best; best = np.where(upd, val, best); scale = np.where(upd, sqx / np.where(sq2 > 0, sq2, 1.0), scale)
    d = scale.astype(np.float16); df = d.astype(np.float32)
    Li = np.searchsorted(MID, x / np.where(df != 0, df, 1.0)[:, None])
    return np.where(ok, d, np.float16(0)), np.where(ok[:, None], NL[Li], 0).astype(np.int8)

