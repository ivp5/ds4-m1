#ifndef DS4_HELP_H
#define DS4_HELP_H

#include <stdio.h>

typedef enum {
    DS4_HELP_DS4,
    DS4_HELP_SERVER,
    DS4_HELP_AGENT,
    DS4_HELP_BENCH,
    DS4_HELP_EVAL,
} ds4_help_tool;

void ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic);

#include "ds4.h"
void llms_adopted_engine_defaults(ds4_engine_options *e);   /* llms ADOPTED (apply_adopted.py) */

#endif
