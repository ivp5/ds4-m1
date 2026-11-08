# Evidence index

Generated index; original evidence is preserved.

- `argsort.metal`: retained source, result or execution artifact.
- `bin.metal`: retained source, result or execution artifact.
- `concat.metal`: retained source, result or execution artifact.
- `cpy.metal`: retained source, result or execution artifact.
- `deepseek4_vision.metal`: retained source, result or execution artifact.
- `dense.metal`: retained source, result or execution artifact.
- `dsv41_plain.metal`: retained source, result or execution artifact.
- `dsv41_plain.metal.1792678719.dsr.bak`: retained source, result or execution artifact.
- `dsv41_ropetab.metal`: retained source, result or execution artifact.
- `dsv41_ropetab.metal.1792685416.dsr.bak`: retained source, result or execution artifact.
- `dsv41_ropetab.metal.bak`: retained source, result or execution artifact.
- `dsv4_hc.metal`: retained source, result or execution artifact.
- `dsv4_kv.metal`: retained source, result or execution artifact.
- `dsv4_kv.metal.1792204236.bak`: retained source, result or execution artifact.
- `dsv4_misc.metal`: retained source, result or execution artifact.
- `dsv4_misc.metal.pre_merge`: retained source, result or execution artifact.
- `dsv4_rope.metal`: retained source, result or execution artifact.
- `flash_attn.metal`: retained source, result or execution artifact.
- `flash_attn.metal.1791923969.bak`: retained source, result or execution artifact.
- `flash_attn.metal.1791972507.bak`: retained source, result or execution artifact.
- `generate_mxfp4_half_lut.py`: retained source, result or execution artifact.
- `get_rows.metal`: retained source, result or execution artifact.
- `glm53_bf16.metal`: retained source, result or execution artifact.
- `glm53_kda.metal`: retained source, result or execution artifact.
- `glm53_vision.metal`: retained source, result or execution artifact.
- `glu.metal`: retained source, result or execution artifact.
- `glu.metal.pre_swbf16`: retained source, result or execution artifact.
- `moe.metal`: retained source, result or execution artifact.
- `norm.metal`: retained source, result or execution artifact.
- `qwen4.metal`: retained source, result or execution artifact.
- `qwen4_vision.metal`: retained source, result or execution artifact.
- `repeat.metal`: retained source, result or execution artifact.
- `set_rows.metal`: retained source, result or execution artifact.
- `softmax.metal`: retained source, result or execution artifact.
- `sum_rows.metal`: retained source, result or execution artifact.
- `unary.metal`: retained source, result or execution artifact.
- `README.md`: directory index.

- dsv41_plain.metal -- the DSV41 library WITHOUT the rope table: NOT what production compiles (production: dsv41_ropetab.metal, set by ds4_help.c and the engine's built-in default since 10-26). A kernel added here is 'not found' at run time.
