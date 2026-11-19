#ifndef LLMS_CONTINUATION_INDEX_H
#define LLMS_CONTINUATION_INDEX_H
/* Request-local committed tokens only. No target imports, globals, threads,
 * per-token allocations, eviction or probabilistic key equality.
 * Source policy: longest level; latest full occurrence, otherwise earliest short.
 * A copied source is selected at the declared lookahead, even for a shorter query.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

enum { CI_OK=0, CI_INVALID=-1, CI_BUDGET=-2, CI_ALLOC=-3, CI_FULL=-4 };
typedef struct { uint64_t a, b; } ci_hash;
typedef struct {
    ci_hash hash;
    uint32_t first, full; /* zero means unused/no mature occurrence; ends >= 1 */
} ci_key;
typedef struct { size_t slot; uint32_t end; } ci_wait;
typedef struct {
    uint32_t length;
    ci_hash power;
    ci_key *keys;
    ci_wait *waiting;
} ci_level;
typedef struct {
    uint32_t capacity, lookahead, length, ring_length;
    size_t level_count, table_slots, allocation_bytes;
    uint32_t *tokens;
    ci_hash *prefix;
    ci_level *levels;
    uint64_t key_comparisons;
    int force_collisions; /* conformance control, normally zero */
} ci_index;
typedef struct {
    const uint32_t *tokens; /* borrowed until reset/truncate/destroy */
    uint32_t count, source_end, level;
} ci_proposal;

static int ci_add_size(size_t *sum, size_t n, size_t width) {
    if (n && width > SIZE_MAX/n) return CI_INVALID;
    size_t bytes=n*width;
    if (*sum > SIZE_MAX-bytes) return CI_INVALID;
    *sum+=bytes;
    return CI_OK;
}

/* Allocation demand, not malloc metadata or an empirical resident-memory claim. */
static int ci_requirements(uint32_t capacity, size_t level_count,
                           uint32_t lookahead, size_t *bytes, size_t *slots) {
    if (!capacity || capacity==UINT32_MAX || !level_count ||
        level_count>capacity || !lookahead || !bytes || !slots)
        return CI_INVALID;
    size_t target=capacity;
    if (target>SIZE_MAX/2) return CI_INVALID;
    target*=2;
    size_t table=1;
    while (table<target) {
        if (table>SIZE_MAX/2) return CI_INVALID;
        table*=2;
    }
    size_t per=0, total=sizeof(ci_index);
    size_t ring=lookahead<capacity ? lookahead : capacity;
    if (ci_add_size(&per,table,sizeof(ci_key)) ||
        ci_add_size(&per,ring,sizeof(ci_wait)) ||
        ci_add_size(&total,level_count,per) ||
        ci_add_size(&total,level_count,sizeof(ci_level)) ||
        ci_add_size(&total,capacity,sizeof(uint32_t)) ||
        ci_add_size(&total,(size_t)capacity+1,sizeof(ci_hash))) return CI_INVALID;
    *bytes=total; *slots=table;
    return CI_OK;
}

static uint64_t ci_power(uint64_t base, uint32_t exponent) {
    uint64_t value=1;
    while (exponent) {
        if (exponent&1) value*=base;
        base*=base; exponent>>=1;
    }
    return value;
}

static void ci_destroy(ci_index *x) {
    if (!x) return;
    if (x->levels) for (size_t j=0;j<x->level_count;j++) {
        free(x->levels[j].keys); free(x->levels[j].waiting);
    }
    free(x->levels); free(x->tokens); free(x->prefix); free(x);
}

static int ci_create(ci_index **out, uint32_t capacity,
                     const uint32_t *lengths, size_t count, uint32_t lookahead,
                     size_t allocation_limit, int force_collisions) {
    if (!out) return CI_INVALID;
    *out=NULL;
    size_t bytes=0, slots=0;
    if (!lengths || ci_requirements(capacity,count,lookahead,&bytes,&slots))
        return CI_INVALID;
    for (size_t j=0;j<count;j++)
        if (!lengths[j] || lengths[j]>=capacity || (j && lengths[j]<=lengths[j-1]))
            return CI_INVALID;
    if (bytes>allocation_limit) return CI_BUDGET;
    ci_index *x=(ci_index *)calloc(1,sizeof(*x));
    if (!x) return CI_ALLOC;
    x->capacity=capacity; x->lookahead=lookahead; x->table_slots=slots;
    x->level_count=count; x->allocation_bytes=bytes;
    x->ring_length=lookahead<capacity ? lookahead : capacity;
    x->force_collisions=force_collisions!=0;
    x->levels=(ci_level *)calloc(count,sizeof(ci_level));
    x->tokens=(uint32_t *)calloc(capacity,sizeof(uint32_t));
    x->prefix=(ci_hash *)calloc((size_t)capacity+1,sizeof(ci_hash));
    if (!x->levels || !x->tokens || !x->prefix) {ci_destroy(x);return CI_ALLOC;}
    for (size_t j=0;j<count;j++) {
        ci_level *l=&x->levels[j];l->length=lengths[j];
        l->power.a=ci_power(UINT64_C(1099511628211),l->length);
        l->power.b=ci_power(UINT64_C(14029467366897019727),l->length);
        l->keys=(ci_key *)calloc(slots,sizeof(ci_key));
        l->waiting=(ci_wait *)calloc(x->ring_length,sizeof(ci_wait));
        if (!l->keys || !l->waiting) {ci_destroy(x);return CI_ALLOC;}
    }
    *out=x;
    return CI_OK;
}

static size_t ci_find(ci_index *x, ci_level *l, uint32_t end) {
    uint32_t n=l->length;
    ci_hash h={0,0};
    if (!x->force_collisions) {
        h.a=x->prefix[end].a-x->prefix[end-n].a*l->power.a;
        h.b=x->prefix[end].b-x->prefix[end-n].b*l->power.b;
    }
    uint64_t mix=h.a^(h.b+(h.a<<6)+(h.a>>2));
    size_t slot=(size_t)mix&(x->table_slots-1);
    for (;;) {
        ci_key *k=&l->keys[slot];
        if (!k->first) return slot;
        if (k->hash.a==h.a && k->hash.b==h.b) {
            x->key_comparisons++;
            if (!memcmp(x->tokens+k->first-n,x->tokens+end-n,(size_t)n*sizeof(uint32_t)))
                return slot;
        }
        slot=(slot+1)&(x->table_slots-1);
    }
}

static int ci_append(ci_index *x, uint32_t token) {
    if (!x) return CI_INVALID;
    if (x->length==x->capacity) return CI_FULL;
    uint32_t old=x->length, p=old+1, end=old;
    x->tokens[old]=token;
    x->prefix[p].a=x->prefix[old].a*UINT64_C(1099511628211)+(uint64_t)token+1;
    x->prefix[p].b=x->prefix[old].b*UINT64_C(14029467366897019727)+(uint64_t)token+1;
    x->length=p;
    for (size_t j=0;j<x->level_count;j++) {
        ci_level *l=&x->levels[j];
        if (end>=l->length) {
            size_t slot=ci_find(x,l,end);ci_key *k=&l->keys[slot];
            if (!k->first) {
                k->first=end;
                if (!x->force_collisions) {
                    k->hash.a=x->prefix[end].a-x->prefix[end-l->length].a*l->power.a;
                    k->hash.b=x->prefix[end].b-x->prefix[end-l->length].b*l->power.b;
                }
            }
            ci_wait *w=&l->waiting[end%x->ring_length];w->slot=slot;w->end=end;
        }
        if (p>=x->lookahead && p-x->lookahead>=l->length) {
            uint32_t eligible=p-x->lookahead;
            ci_wait *w=&l->waiting[eligible%x->ring_length];
            /* Each end is inserted at end+1 and matures at end+lookahead.
             * The ring has not overwritten it; no key or table ever moves. */
            l->keys[w->slot].full=eligible;
        }
    }
    return CI_OK;
}

static int ci_propose(ci_index *x, uint32_t limit, ci_proposal *out) {
    if (!x || !out || limit>x->lookahead) return CI_INVALID;
    memset(out,0,sizeof(*out));
    if (!limit) return CI_OK;
    uint32_t p=x->length;
    for (size_t j=x->level_count;j>0;j--) {
        ci_level *l=&x->levels[j-1];
        if (p<=l->length) continue;
        ci_key *k=&l->keys[ci_find(x,l,p)];
        if (!k->first) continue;
        uint32_t end=k->full ? k->full : k->first;
        uint32_t available=p-end;
        out->tokens=x->tokens+end;
        out->count=limit<available ? limit : available;
        out->source_end=end;out->level=l->length;
        return CI_OK;
    }
    return CI_OK;
}

static void ci_reset(ci_index *x) {
    if (!x) return;
    x->length=0; x->prefix[0].a=0; x->prefix[0].b=0;
    for (size_t j=0;j<x->level_count;j++) {
        memset(x->levels[j].keys,0,x->table_slots*sizeof(ci_key));
        memset(x->levels[j].waiting,0,(size_t)x->ring_length*sizeof(ci_wait));
    }
}

/* Exceptional accepted-history rewind rebuilds the index without allocation.
 * Speculative rejected tokens should never have been appended at all. */
static int ci_truncate(ci_index *x, uint32_t length) {
    if (!x || length>x->length) return CI_INVALID;
    if (length==x->length) return CI_OK;
    ci_reset(x);
    for (uint32_t i=0;i<length;i++) {
        uint32_t token=x->tokens[i];
        int rc=ci_append(x,token);
        if (rc) return rc;
    }
    return CI_OK;
}
#endif
