/* llms_diag.h (llms 11-19): ONE GATE FOR EVERY ENGINE DIAGNOSTIC. Our patches print about 2,000 stderr lines that open with "llms": witnesses and counters that the test
 * harness reads from its logs. On a person's terminal they are noise (the user 11-19: 'who are these debug messages for?? ... cull!'). So every fprintf of these programs goes
 * through llms_diag_fprintf (ds4.c): a line to stderr whose format opens with "llms" is printed when stderr is a file or a pipe and DROPPED when stderr is a terminal.
 * A line that announces the process is stopping (FATAL, REFUSING / REFUSED / refusing, an exit code written "(exit N") always prints. LLMS_DS4_DIAG=1 prints everything anywhere; LLMS_DS4_DIAG=0 drops them
 * everywhere. A diagnostic written in pieces is dropped whole: after a dropped piece without a newline, stderr pieces are dropped up to the one that ends the line. */
#ifndef LLMS_DIAG_H
#define LLMS_DIAG_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
int llms_diag_fprintf(FILE *f, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#ifdef __cplusplus
}
#endif
#define fprintf(...) llms_diag_fprintf(__VA_ARGS__)
#endif
