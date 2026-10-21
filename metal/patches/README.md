# Metal patches selected by the optimized profiles

Each `.metal.patch` below reconstructs the named source under
`build/metal/`. `manifest.json` binds its base, patch and result with
SHA-256 hashes. `make metal-variants` generates them automatically during a
normal build and refuses changed bases until the patches are explicitly rebased.
Related variants share one parent patch; generation follows dependency order and
rejects missing parents or cycles. The resulting source hashes are unchanged.
Do not edit the generated files; use `metal/update_kernel.py`.

Every source below is selected by at least one program profile. The engine also
loads the standard upstream sources in `metal/`. Keep these filenames: several
runtime capability decisions depend on their feature suffixes.

| File | Program profiles |
| --- | --- |
| `argsort_shuffle.metal.patch` | agent, cli, server |
| `dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf.metal.patch` | server |
| `dense_llms2t_b_over_dense_llms2s_over_dense_llms2r_mvb_q4l_rbf_dual_k_h2_off32k.metal.patch` | agent, cli |
| `dsv4_hc_llms12_llms20_hcf.metal.patch` | server |
| `dsv4_hc_llms12_llms20_hcf_rows.metal.patch` | agent, cli |
| `dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast.metal.patch` | server |
| `dsv4_misc_sg_llms4_llms11_llms13_llms15_llms16_llms17_llms18_tkfast_mqa4_psel_tkdet_idxc_att2_tkcomb.metal.patch` | agent, cli |
| `moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64_redu.metal.patch` | server |
| `moe_llms_d8lhd8q4x3wcf_ctx_cffb_cfdec10_bs6c_bs64n_redu_colz_xpar_idx32_g32d2m8xj.metal.patch` | agent, cli |
| `norm_mvb.metal.patch` | agent, cli, server |
| `dsv41_ropetab.metal.patch` | agent, cli, server |
