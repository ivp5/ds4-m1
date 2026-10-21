#define main converter_entrypoint
#include "../gguf-tools/deepseek4-quantize.c"
#undef main
#include <assert.h>
int main(void) {
    const char *action=NULL;
    char *n=map_dspark_hf_name("mtp.2.markov_head.embed.weight",&action);
    assert(n && !strcmp(n,"mtp.2.markov_head.markov_w1.weight"));free(n);
    n=map_dspark_hf_name("mtp.2.markov_head.head.weight",&action);
    assert(n && !strcmp(n,"mtp.2.markov_head.markov_w2.weight"));free(n);
    unsigned char weights[128*128],scale[16];memset(weights,0x38,sizeof weights);memset(scale,127,sizeof scale);
    st_value w={0},s={0};
    w.dtype="F8_E4M3";w.n_dims=2;w.shape[0]=128;w.shape[1]=128;w.data=weights;w.nbytes=sizeof weights;
    s.dtype="F8_E8M0";s.n_dims=2;s.shape[0]=4;s.shape[1]=4;s.data=scale;s.nbytes=16;
    int64_t count=0;float *v=dequant_fp8_weight(&w,&s,&count);assert(count==128*128);
    for(int i=0;i<count;i++)assert(v[i]==1.0f);free(v);
    s.shape[0]=s.shape[1]=1;s.nbytes=1;v=dequant_fp8_weight(&w,&s,&count);
    for(int i=0;i<count;i++)assert(v[i]==1.0f);free(v);
    puts("DSpark name mapping and FP8 32/128 scale blocks: PASS");return 0;
}
