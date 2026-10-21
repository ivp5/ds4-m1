# Maintaining the optimized fork

Keep runtime changes in the upstream file layout. Do not export a development
workspace wholesale. Each update should be a reviewable commit containing the
runtime change, its selected profile/patch changes, and its validation limits.

## Sources of truth

| Edit | Regenerate / validate |
| --- | --- |
| Runtime C / Objective-C / canonical `metal/` sources | Normal `make`; upstream layout is retained |
| `ds4_defaults.json` | `python3 gguf-tools/generate_defaults.py`; ignored `ds4_defaults.h` is derived |
| Reviewed full Metal variant | `update_kernel.py` records its patch and manifest; `make` reconstructs it |
| Asset format or quantization | Update both runtime reader and builder, then a roundtrip/corruption check |
| Calibration bundle | Independent data only; replace provenance/checksums and re-evaluate held-out quality |

Only kernels selected by the public profiles belong in `metal/patches/`.
The full generated sources live in ignored `build/metal/`; never
commit or hand-edit them. Preserve their filenames: the engine currently uses
feature suffixes for some capability decisions. The manifest records SHA-256
for the base, patch and complete output. A base can be a canonical source or
another selected variant under build/metal/; update parents before their children. A changed base fails the build until
explicitly rebased; silently applying an approximate patch is forbidden.

To update one reviewed kernel:

```sh
python3 metal/update_kernel.py --name EXACT_EXISTING_NAME.metal --source /path/to/reviewed/source.metal
make metal-variants
```

For a new variant, also supply `--base metal/moe.metal` (or its actual base),
select it in `ds4_defaults.json`, and remove a superseded variant only once no profile
references it. Profile paths use `@repo/`, `@model/` and `@draft/` placeholders;
never use a maintainer's absolute paths. Rebuild after changing profiles.

## Check before committing

On Apple Silicon, with the Python builder requirements installed:

```sh
make -j2 ds4 ds4-agent ds4-server
make test-metal-pack
git diff --check
git diff --stat
```

`test-metal-pack` checks generated defaults, reconstructs kernels, checks their
hashes, runs CPU codec/format tests, compares the compiled profile constructors,
compiles the actual Metal source assemblies offline, and runs converter tests.
It runs no model and establishes no end-to-end throughput or fidelity result.
Before claiming an inference improvement, evaluate independently on at least two
workloads, one held out, with fixed calibration, paired order, output correctness,
and cold/warm timing separated. Retain evidence privately; publish a concise
reproducible result rather than raw prompts, environment dumps or machine paths.

## Merge upstream

The initial base is `5379c1c9ac315f1540371b06a3db88491d4059ff` in antirez/ds4.
Keep the `upstream` remote and use ordinary Git three-way merges. Before merging,
generate the selected kernels and preserve reviewed full variants in an ignored
local directory so they can be compared with changed base sources. Resolve each
runtime conflict, rebase affected variants onto the new canonical sources, then
record them with `update_kernel.py`. Do not merely update the base hash to silence
a failure. Run the checks above and a native model comparison before publishing.

Keep the upstream README below the fork's short setup section intact where
possible; detailed setup belongs in METAL_SETUP.md. Keep changes in their owning file rather than adding copied engines
or version chains. Optional paths remain only where they still support a distinct
profile or a meaningful comparison; remove them with evidence, not by folding
all flags to a snapshot's values.

## Release hygiene

Review the full diff against the upstream base, including comments and binary
assets. No private prompts, usernames, machine paths, credentials, captures,
benchmark training data, journals, weights, or local logs belong in Git. Build products under `build/`, and weight files under `gguf/`, are ignored. Check `git status` before
staging; never use the presence of `.gitignore` as proof an old tracked file is safe.

Commit locally after review. Publishing is a separate explicit action; none of
the build, patch, or validation tools commits or pushes on your behalf.
