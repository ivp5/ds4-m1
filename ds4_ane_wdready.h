/* Pure readiness generation shared by the proposed engine patch and CPU controls. */
#ifndef LLMS_ANE_WDREADY_H
#define LLMS_ANE_WDREADY_H
#include <stdint.h>
static inline int llms_ane_handoff_ready(uint64_t ready, uint64_t completed) {
    return ready > completed;
}
static inline void llms_ane_ready_publish(uint64_t *value, uint64_t target) {
    uint64_t previous = __atomic_load_n(value, __ATOMIC_RELAXED);
    while (previous < target &&
           !__atomic_compare_exchange_n(value, &previous, target, 1,
                                        __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
}
#endif
