#ifndef IMITSAT_CLIENT_H_INCLUDED
#define IMITSAT_CLIENT_H_INCLUDED

#include <stddef.h>

/* Limits for the dynamic prefix we send to ImitSAT. */
#define IMITSAT_MAX_DECISIONS   4096
#define IMITSAT_DYN_CAP         8192

typedef struct imitsat_options {
  int  imitsat_enabled;              /* 0/1 */
  char imitsat_host[64];             /* "127.0.0.1" etc */
  int  imitsat_port;                 /* TCP port (e.g. 8765) */
  int  imitsat_limit;                /* max guided decisions (0 = unlimited) */
} imitsat_options;

/* Per-instance TCP client state stored in 'struct kissat'. */
typedef struct imitsat_client {
  int  fd;                     /* socket fd, -1 if closed */
  int  timeout_ms;             /* socket timeout in ms */

  int  guide_limit;            /* hard cap on guided decisions (0 = unlimited) */
  int  guided_used;            /* how many guided decisions we actually used */

  char last_error[128];        /* last error message (optional) */

  /* ImitSAT dynamic context = list of accepted decisions (signed DIMACS). */
  int      decisions[IMITSAT_MAX_DECISIONS];
  unsigned num_decisions;

  /* Cached string buffer for dyn_text (" D d1 D d2 ... D"). */
  char dyn_buf[IMITSAT_DYN_CAP];
} imitsat_client;

/* Socket + JSON protocol. */
int  imitsat_connect       (imitsat_client *c, const char *host, int port);
void imitsat_close         (imitsat_client *c);
int  imitsat_send_hello    (imitsat_client *c, const char *cnf_dimacs);
int  imitsat_next_decision (imitsat_client *c, const char *dyn_text, int *out_lit);

/* Dyn helpers: manage the decision list and build the text " D d1 D d2 ... D". */

/* Reset to "no decisions yet".  Next dyn string will be " D". */
void        imitsat_dyn_reset         (imitsat_client *c);

/* Record a new accepted decision (ext_lit is DIMACS ±var). */
void        imitsat_dyn_note_decision (imitsat_client *c, int ext_lit);

/* Return " D" (no decisions) or " D d1 D d2 ... D" (with trailing ' D'). */
const char *imitsat_dyn_cstr          (imitsat_client *c);

#endif /* IMITSAT_CLIENT_H_INCLUDED */
