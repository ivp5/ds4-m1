# Export validation

Validated on Apple Silicon macOS, 2026-10-21, after integration into the upstream source layout:

- CLI, agent and server compile and link with the current toolchain.
- All three effective profiles match the compiled constructor; native launches default to the optimized profile on Apple Silicon, explicit overrides win, and upstream mode installs no optimized defaults.
- All three distinct 26-file Metal source assemblies compile offline. A deliberately invalid statement fails compilation. Agent and CLI use the same source assembly.
- Nine CPU tests pass: generated-parent ordering and cycle/missing-parent refusal, hash-checked kernel reconstruction with base/patch/output corruption guards, GGUF bounds, Q4L packing/zero blocks, Engram zero blocks, BS64 small blocks, bank selection, ridge normal equations, and agent trace transitions.
- The C converter's synthetic tests cover FP8 32/128 scale grids and revised prediction-head names.
- The converter's dry-run on the prediction source reports 78 tensors and 4,500,523,744 output bytes.
- Metadata plans on a compatible main GGUF report BS64 99,090,448,384 bytes, G32 53,791,965,184 bytes, dense Q4L 323 tensors / 4,194,385,920 bytes.
- G32 C encoder/verifier compile. Bundle checksums and the generic bank's reconstruction from its five count tables pass.
- The current upstream Q2 download header has matching tokenizer metadata and all 1,046 tensor entries; the full payload was not rehashed or run.
- Offline downloader fixture passes revision pinning, checksum refusal, verified-file reuse and symlink checks.
- All 11 generated Metal variants are byte-identical to the complete reviewed export sources.
- Native startup checks report individual missing or empty selected assets and the setup guide; metadata/receipt validation is available separately with `gguf-tools/assets.py`. Source/privacy scan and Git whitespace checks pass.

Not established: full regenerated-pack byte identity, GPU runtime pipeline
specializations, native output parity, long-context quality, calibrated
lookahead quality, or end-to-end speed/fidelity for the exported pack. These
checks do not turn historical runtime measurements into a result for this fork.
The generic BS64 calibration and bank deliberately differ from historical
benchmark-derived data. Holdout evaluation is required before performance claims.

Run `make test-metal-pack` for the model-free checks. Build/test logs,
compiler products and captures stay under ignored `build/`.
