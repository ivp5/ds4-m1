/* Per-user namespace, shared by all copies of this fork. */
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
static char llms_shm_names[5][32];
static pthread_once_t llms_shm_once = PTHREAD_ONCE_INIT;
static void llms_shm_init_names(void) {
    const unsigned u=(unsigned)getuid();
    snprintf(llms_shm_names[4],32,"/d4.%u.s.0",u);
    snprintf(llms_shm_names[0],32,"/d4.%u.hdr",u);
    snprintf(llms_shm_names[1],32,"/d4.%u.bank",u);
    snprintf(llms_shm_names[2],32,"/d4.%u.s.%%d",u);
    snprintf(llms_shm_names[3],32,"/d4.%u.s.%%u",u);
}
static const char *llms_shm_name(int k) {
    pthread_once(&llms_shm_once,llms_shm_init_names); return llms_shm_names[k];
}
