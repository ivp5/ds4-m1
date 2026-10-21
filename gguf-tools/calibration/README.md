# Bundled calibration aggregates

This directory is shipped with the fork. No Hugging Face account or additional
calibration download is required. The data files total 10,941,943 bytes before
this README. Model weight overlays are generated separately.

| File | Bytes | Meaning |
| --- | ---: | --- |
| `dense_means.npz` | 9,245,233 | 321 named float32 vectors of per-input-column mean square activation |
| `importance.npy` | 1,638,528 | Float64 array [40,5120], derived from the shared FFN input statistics |
| `manifest.json` | 34,690 | Source checkpoint, model header identity, calibration corpus hash, counts, tensor shapes and SHA-256 checksums |
| `bank.txt` | 402 | Three rows of allowed prediction-stage expert IDs |
| `bank.json` | 22,193 | All five calibration count tables, selection rule and hashes |
| `bank_prompts.json` | 897 | Five generic authored prompts used for those counts |

The dense aggregate source was collected on separate prose excerpts from
Popper (1959), Aumann and Hart (volume 3), O'Donnell (*Analysis of Boolean
Functions*), and Norton (2021). Those excerpts and model-generated responses
are not redistributed. The recorded calibration construction checked overlap
against evaluation texts. These aggregate values are reused; the original
collection was not rerun during export.

The legacy dense statistics file used byte offsets. This package identifies
vectors by tensor name and validates tensor dimensions, so changing GGUF metadata
or tensor ordering cannot silently attach a vector to a different matrix.
The values are identical to the normalized source statistics. They are intended
for DeepSeek-V4.1-Flash-1017; using another checkpoint requires new calibration.
A header fingerprint identifies the original GGUF directory, not every weight
byte. The manifest must not be interpreted as a full checkpoint-content digest.

Why reuse shared-input statistics for BS64? The decoder's shared and routed
FFN gate/up projections consume the same normalized FFN input. The shared gate
and up statistics agree at all 40 layers. Therefore their per-column second
moment supplies the input weighting required by the BS64 encoder, without
another full activation capture. This table is a **new, independently calibrated
replacement**, not the benchmark-derived importance used for historical speed
measurements. Neither its model fidelity nor resulting speed has been measured
in the exported runtime. Test on independent held-out tasks before adopting.

The bundled drafter bank is likewise built from the five generic prompts, not
from evaluation-task routes. Its counts are sufficient to reproduce the ID
selection with `build_bank.choose`. This can change acceptance relative to a
historical benchmark-derived bank. A faster bank on its own training task is
not evidence of general improvement.

`../calibration_prompts.json` provides eight additional generic prompts for new
collection through `../calibrate.py`. They are a reproducible starter
corpus, not proof of broad calibration coverage. Lookahead fitting still needs
aligned, per-layer captures; those larger files are not included. Neither dense
moments nor the importance table contains the cross-layer observations needed
to fit lookahead. Keep all calibration inputs disjoint from evaluation inputs.
