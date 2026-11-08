/* G16D v1: bounded virtual records over an immutable G32 base and changed bytes.
 * No new worker, lock, or disk image. Included by an off-by-default engine switch.
 * Partial callers receive bytes only after the complete record passes its binding.
 */
#ifndef LLMS_G16_DELTA_READER_H
#define LLMS_G16_DELTA_READER_H
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { G16D_OLD=3502080, G16D_NEW=3870720, G16D_Q=2949120,
       G16D_S32=368640, G16D_S16=737280, G16D_MASK=483840,
       G16D_PREFIX=72, G16D_HEADER=4096, G16D_LAYERS=40, G16D_EXPERTS=384 };
/* Selected once, before workers or thread-local record caches exist. */
static size_t g16d_new_bytes=G16D_NEW, g16d_mask_bytes=G16D_MASK, g16d_codes_bytes=G16D_S16;
static unsigned g16d_group=16;
#include "ds4_g16_scatter.h"
static int g16d_simd; /* LLMS_DS4_G16_DELTA_SIMD; initialized before reader workers; OFF unless set. */

static int g16d_active, g16d_base_fd=-1, g16d_fd[G16D_LAYERS];
static uint64_t g16d_offsets[G16D_LAYERS][G16D_EXPERTS+1], g16d_base_data, g16d_identity;
static uint64_t g16d_reads, g16d_records, g16d_disk_bytes;
/* LLMS_DS4_DELTA_BASE_ONCE; OFF unless set before reader workers exist.
 * Validation status only: no model bytes, selected routes or outputs stored. */
static int g16d_base_once, g16d_changed;
static unsigned char g16d_validated[G16D_LAYERS][G16D_EXPERTS];
static struct stat g16d_base_stat, g16d_file_stat[G16D_LAYERS];
static int g16d_same_stat(const struct stat *a,const struct stat *b) {
    return a->st_dev==b->st_dev && a->st_ino==b->st_ino && a->st_size==b->st_size &&
        a->st_mtimespec.tv_sec==b->st_mtimespec.tv_sec &&
        a->st_mtimespec.tv_nsec==b->st_mtimespec.tv_nsec &&
        a->st_ctimespec.tv_sec==b->st_ctimespec.tv_sec &&
        a->st_ctimespec.tv_nsec==b->st_ctimespec.tv_nsec;
}
static int g16d_unchanged(unsigned L) {
    if(!g16d_base_once)return 1;
    if(L>=G16D_LAYERS || __atomic_load_n(&g16d_changed,__ATOMIC_ACQUIRE))return 0;
    struct stat b,p;
    if(fstat(g16d_base_fd,&b) || fstat(g16d_fd[L],&p) ||
       !g16d_same_stat(&b,&g16d_base_stat) || !g16d_same_stat(&p,&g16d_file_stat[L])) {
        __atomic_store_n(&g16d_changed,1,__ATOMIC_RELEASE);
        fprintf(stderr,"llms delta: immutable descriptor changed; all further reads refused\n");return 0;
    }
    return 1;
}

static uint32_t g16d_u32(const void *p) { uint32_t n; memcpy(&n,p,4); return n; }
static uint64_t g16d_u64(const void *p) { uint64_t n; memcpy(&n,p,8); return n; }
static int g16d_exact(int fd, void *buf, size_t n, uint64_t at) {
    if(at>(uint64_t)LLONG_MAX || n>(uint64_t)LLONG_MAX-at) return 0;
    size_t done=0;
    while(done<n) { ssize_t got=pread(fd,(uint8_t *)buf+done,n-done,(off_t)(at+done));
        if(got<0 && errno==EINTR) continue;
        if(got<=0) return 0;
        done+=(size_t)got;
    }
    return 1;
}
static void g16d_sha(const void *p, size_t n, uint8_t out[32]) {
    (void)CC_SHA256(p,(CC_LONG)n,out);
}
static int g16d_expand_checked(const uint8_t *base, const uint8_t *patch, size_t pn, uint8_t *out, int base_checked) {
    if(!base || !patch || !out || pn<G16D_PREFIX) return 0;
    const uint32_t count=g16d_u32(patch+64), flags=g16d_u32(patch+68);
    if(flags || count>g16d_new_bytes || pn!=(size_t)G16D_PREFIX+(count ? (size_t)g16d_mask_bytes+count : 0)) return 0;
    uint8_t digest[32];
    if(!base_checked) {
        g16d_sha(base,G16D_OLD,digest);
        if(memcmp(digest,patch,32)) return 0;
    }
    memcpy(out,base,G16D_Q);
    if(g16d_group==8) {
        for(size_t i=0;i<G16D_S32;i++) out[G16D_Q+4*i]=out[G16D_Q+4*i+1]=out[G16D_Q+4*i+2]=out[G16D_Q+4*i+3]=base[G16D_Q+i];
    } else for(size_t i=0;i<G16D_S32;i++) out[G16D_Q+2*i]=out[G16D_Q+2*i+1]=base[G16D_Q+i];
    memcpy(out+G16D_Q+g16d_codes_bytes,base+G16D_Q+G16D_S32,G16D_OLD-G16D_Q-G16D_S32);
    if(count) {
        const uint8_t *mask=patch+G16D_PREFIX, *vals=mask+g16d_mask_bytes;
        uint32_t used=0;
        if(g16d_simd) {
            if(!g16d_scatter_simd(out,mask,g16d_mask_bytes,vals,count))return 0;
            used=count;
        } else for(size_t i=0;i<g16d_mask_bytes;i++) {
            unsigned bits=mask[i];
            while(bits) { const unsigned b=(unsigned)__builtin_ctz(bits); bits&=bits-1;
                if(used>=count) return 0;
                out[8*i+b]=vals[used++];
            }
        }
        if(used!=count) return 0;
    }
    g16d_sha(out,g16d_new_bytes,digest);
    return memcmp(digest,patch+32,32)==0;
}
static int g16d_expand(const uint8_t *base, const uint8_t *patch, size_t pn, uint8_t *out) {
    return g16d_expand_checked(base,patch,pn,out,0);
}
static uint64_t g16d_mix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b=p;
    for(size_t i=0;i<n;i++) { h^=b[i]; h*=1099511628211ull; }
    return h;
}
static void g16d_report(void) {
    fprintf(stderr,g16d_group==8 ? "llms g8 delta: %llu ranges, %llu bound records reconstructed, %.3f GB read; bounded thread-local scratch\n" : "llms g16 delta: %llu ranges, %llu bound records reconstructed, %.3f GB read; bounded thread-local scratch\n",
        (unsigned long long)g16d_reads,(unsigned long long)g16d_records,(double)g16d_disk_bytes/1e9);
}
static int g16d_source_group(const char *src) {
    if(!src || !*src) return 0;
    int fd=open(src,O_RDONLY|O_CLOEXEC); if(fd<0) return 0;
    char h[256]={0}; ssize_t n=pread(fd,h,sizeof(h)-1,0); close(fd);
    if(n<=0)return 0;
    const int a=strstr(h,"#define LLMS_G16D_LAYOUT_V1 1")!=NULL;
    const int b=strstr(h,"#define LLMS_G8D_LAYOUT_V1 1")!=NULL;
    return a&&b ? -1 : b ? 8 : a ? 16 : 0;
}
static int g16d_choose_group(unsigned group) {
    if(group!=8 && group!=16)return 0;
    if(g16d_active)return group==g16d_group;
    g16d_group=group;
    g16d_codes_bytes=G16D_S32*(32u/group);
    g16d_new_bytes=G16D_Q+g16d_codes_bytes+(G16D_OLD-G16D_Q-G16D_S32);
    g16d_mask_bytes=g16d_new_bytes/8u;
    return 1;
}
static int g16d_open(const char *dir, int base_fd, uint64_t data) {
    if(!dir || !*dir || !strcmp(dir,"0")) return 0;
    const char *simd=getenv("LLMS_DS4_G16_DELTA_SIMD");g16d_simd=simd&&*simd&&strcmp(simd,"0");
    if(g16d_simd)fprintf(stderr,"llms g16 delta: exact SIMD byte scatter ON (both SHA256 bindings retained)\n");
    const char *once=getenv("LLMS_DS4_DELTA_BASE_ONCE");g16d_base_once=once&&*once&&strcmp(once,"0");
    if(g16d_base_once) {
        if(g16d_active || __atomic_load_n(&g16d_changed,__ATOMIC_ACQUIRE) || fstat(base_fd,&g16d_base_stat))return 0;
        fprintf(stderr,"llms delta: base SHA once per record; target SHA every reconstruction; immutable FD checks ON\n");
    }
    uint64_t ident=1469598103934665603ull;
    for(unsigned L=0;L<G16D_LAYERS;L++) g16d_fd[L]=-1;
    for(unsigned L=0;L<G16D_LAYERS;L++) {
        char path[PATH_MAX]; uint8_t h[G16D_HEADER]; struct stat st;
        int z=snprintf(path,sizeof path,"%s/D%02u.%s",dir,L,g16d_group==8 ? "g8d" : "g16d");
        if(z<0 || (size_t)z>=sizeof path) goto bad;
        int fd=open(path,O_RDONLY|O_CLOEXEC); if(fd<0) goto bad; g16d_fd[L]=fd;
        if(fstat(fd,&st) || !g16d_exact(fd,h,sizeof h,0) || memcmp(h,g16d_group==8 ? "G08L" : "G16L",4) ||
           g16d_u32(h+4)!=1 || g16d_u32(h+8)!=L || g16d_u32(h+12)!=G16D_EXPERTS ||
           g16d_u32(h+16)!=5120 || g16d_u32(h+20)!=2304 ||
           g16d_u32(h+24)!=G16D_OLD || g16d_u32(h+28)!=g16d_new_bytes) goto bad;
        for(unsigned e=0;e<=G16D_EXPERTS;e++) g16d_offsets[L][e]=g16d_u64(h+64+8*e);
        if(g16d_offsets[L][0]!=G16D_HEADER || st.st_size<0 ||
           g16d_offsets[L][G16D_EXPERTS]!=(uint64_t)st.st_size) goto bad;
        for(unsigned e=0;e<G16D_EXPERTS;e++) {
            uint64_t a=g16d_offsets[L][e],b=g16d_offsets[L][e+1];
            if(b<a+G16D_PREFIX || b-a>G16D_PREFIX+g16d_mask_bytes+g16d_new_bytes) goto bad;
        }
        if(g16d_base_once)g16d_file_stat[L]=st;
        const uint64_t meta[]={ (uint64_t)st.st_ino,(uint64_t)st.st_size,
            (uint64_t)st.st_mtimespec.tv_sec,(uint64_t)st.st_mtimespec.tv_nsec };
        ident=g16d_mix(ident,meta,sizeof meta); ident=g16d_mix(ident,h,sizeof h);
        (void)fcntl(fd,F_NOCACHE,1); (void)fcntl(fd,F_RDAHEAD,0);
    }
    g16d_base_fd=base_fd;
    if(g16d_base_once)for(unsigned L=0;L<G16D_LAYERS;L++)if(!g16d_unchanged(L))goto bad;
    g16d_base_data=data; g16d_identity=ident; g16d_active=1;
    atexit(g16d_report);
    fprintf(stderr,"llms: group%u down delta (%s; G32 base bound per record; %zu-byte expanded down; immutable identity %016llx)\n",g16d_group,dir,g16d_new_bytes,(unsigned long long)ident);
    return 1;
bad:
    for(unsigned L=0;L<G16D_LAYERS;L++) if(g16d_fd[L]>=0) { close(g16d_fd[L]); g16d_fd[L]=-1; }
    fprintf(stderr,"llms g16 delta: invalid or missing layer/header in %s; refusing mixed layout\n",dir);
    return 0;
}
static int g16d_read(uint64_t off, uint64_t n, uint8_t *dst, uint64_t *disk_bytes) {
    if(disk_bytes) *disk_bytes=0;
    const uint64_t total=(uint64_t)G16D_LAYERS*G16D_EXPERTS*g16d_new_bytes;
    if(!g16d_active || !dst || off<g16d_base_data || off-g16d_base_data>total || n>total-(off-g16d_base_data)) return 0;
    static __thread uint8_t *base,*target,*patch;
    static __thread uint64_t last=UINT64_MAX;
    if(!base) {
        base=malloc(G16D_OLD); target=malloc(g16d_new_bytes); patch=malloc(G16D_PREFIX+g16d_mask_bytes+g16d_new_bytes);
        if(!base || !target || !patch) { free(base);free(target);free(patch);base=target=patch=NULL;return 0; }
    }
    uint64_t pos=off-g16d_base_data, used=0;
    while(n) {
        uint64_t rec=pos/g16d_new_bytes, inner=pos%g16d_new_bytes, take=g16d_new_bytes-inner;
        if(take>n) take=n;
        unsigned L=(unsigned)(rec/G16D_EXPERTS),e=(unsigned)(rec%G16D_EXPERTS);
        if(!g16d_unchanged(L))return 0;
        if(rec!=last) {
            last=UINT64_MAX;
            uint64_t p0=g16d_offsets[L][e],pn=g16d_offsets[L][e+1]-p0;
            if(!g16d_exact(g16d_base_fd,base,G16D_OLD,g16d_base_data+rec*G16D_OLD) ||
               !g16d_exact(g16d_fd[L],patch,(size_t)pn,p0) || !g16d_expand_checked(base,patch,(size_t)pn,target,g16d_base_once && __atomic_load_n(&g16d_validated[L][e],__ATOMIC_ACQUIRE))) {
                fprintf(stderr,"llms g16 delta: layer %u expert %u read/binding failure\n",L,e); return 0;
            }
            used+=G16D_OLD+pn; last=rec;
            __sync_fetch_and_add(&g16d_records,1);
        }
        memcpy(dst,target+inner,(size_t)take);
        if(!g16d_unchanged(L))return 0;
        if(g16d_base_once)__atomic_store_n(&g16d_validated[L][e],1,__ATOMIC_RELEASE);
        dst+=take;pos+=take;n-=take;
    }
    if(disk_bytes) *disk_bytes=used;
    __sync_fetch_and_add(&g16d_reads,1); __sync_fetch_and_add(&g16d_disk_bytes,used);
    return 1;
}
#endif
