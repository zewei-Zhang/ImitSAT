#include "application.h"
#include "cover.h"
#include "handle.h"
#include "kissat.h"
#include "print.h"

#include <assert.h>
#include <stdbool.h>

#include "options.h"          // still fine to include, but we no longer use kissat_get_option here
#include "internal.h"         // access solver internals (imopts, imc)
#include "imitsat_client.h"   // TCP client

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static kissat *volatile solver;

// clang-format off
static void
kissat_signal_handler (int sig)
{
  kissat_signal (solver, "caught", sig);
  kissat_print_statistics (solver);
  kissat_signal (solver, "raising", sig);
#ifdef QUIET
  (void) sig;
#endif
  FLUSH_COVERAGE (); } 

static volatile bool ignore_alarm = false;

static void
kissat_alarm_handler (void)
{
  if (ignore_alarm)
    return;
  assert (solver);
  kissat_terminate (solver);
}

#ifndef NDEBUG
extern int kissat_dump (kissat *);
#endif

#include "error.h"
#include "random.h"
#include <strings.h>

/* ---------------- ImitSAT helpers (only used in this file) ---------------- */

/* Parse ImitSAT-related options directly from argv, before kissat_application()
   parses its own options. This avoids relying on kissat_get_option, which only
   sees values after the internal parser runs. */
static void
imitsat_init_from_argv (kissat *s, int argc, char **argv)
{
  const char *host = getenv ("IMITSAT_HOST");
  if (!host || !*host)
    host = "127.0.0.1";

  /* Defaults matching options.h */
  s->imopts.imitsat_enabled = 0;
  snprintf (s->imopts.imitsat_host,
            sizeof s->imopts.imitsat_host,
            "%s", host);
  s->imopts.imitsat_port  = 8765;
  s->imopts.imitsat_limit = 0;

  s->imc.fd          = -1;
  s->imc.timeout_ms  = 200;   // default imitsat_timeout in options.h
  s->imc.guide_limit = 0;
  s->imc.guided_used = 0;

  /* Parse command-line arguments of the form:
       --imitsat[=0|1]
       --imitsat_limit=<int>
       --imitsat_port=<int>
       --imitsat_timeout=<ms>
   */
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!a || a[0] != '-')
      continue;

    if (!strncmp (a, "--imitsat=", 10)) {
      s->imopts.imitsat_enabled = (atoi (a + 10) != 0);
    } else if (!strcmp (a, "--imitsat")) {
      s->imopts.imitsat_enabled = 1;
    } else if (!strncmp (a, "--imitsat_limit=", 16)) {
      int v = atoi (a + 16);
      if (v >= 0)
        s->imopts.imitsat_limit = v;
    } else if (!strncmp (a, "--imitsat_port=", 15)) {
      int p = atoi (a + 15);
      if (p > 0 && p < 65536)
        s->imopts.imitsat_port = p;
    } else if (!strncmp (a, "--imitsat_timeout=", 18)) {
      int t = atoi (a + 18);
      if (t >= 0)
        s->imc.timeout_ms = t;
    }
  }

  /* Guide limit equals the configured imitsat_limit.  If it is 0,
     the hook in decide.c will effectively never ask the model. */
  s->imc.guide_limit = s->imopts.imitsat_limit;

  /* Clear dynamic decision tail buffer (fields are added in internal.h). */
  s->imitsat_dyn_sz = 0;
  s->imitsat_dyn[0] = 0;
}

static const char *
find_input_path (int argc, char **argv)
{
  // Heuristic: last non-option argument
  const char *path = NULL;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!a || !*a)
      continue;
    if (a[0] == '-')
      continue;
    path = a;
  }
  return path;
}

static char *
read_entire_file (const char *path)
{
  if (!path)
    return NULL;
  FILE *fp = fopen (path, "rb");
  if (!fp)
    return NULL;
  if (fseek (fp, 0, SEEK_END)) {
    fclose (fp);
    return NULL;
  }
  long n = ftell (fp);
  if (n < 0) {
    fclose (fp);
    return NULL;
  }
  if (fseek (fp, 0, SEEK_SET)) {
    fclose (fp);
    return NULL;
  }
  char *buf = (char *) malloc ((size_t) n + 1);
  if (!buf) {
    fclose (fp);
    return NULL;
  }
  size_t rd = fread (buf, 1, (size_t) n, fp);
  fclose (fp);
  if (rd != (size_t) n) {
    free (buf);
    return NULL;
  }
  buf[n] = 0;
  return buf;
}

/* ------------------------------------------------------------------------- */

int
main (int argc, char **argv)
{
  int res;
  solver = kissat_init ();
  kissat_init_alarm (kissat_alarm_handler);
  kissat_init_signal_handler (kissat_signal_handler);

  /* ---- ImitSAT: read argv and (optionally) connect & send 'hello' ---- */
  imitsat_init_from_argv (solver, argc, argv);

  if (solver->imopts.imitsat_enabled) {
    const char *path = find_input_path (argc, argv);
    char *dimacs = read_entire_file (path);     // for stdin this will be NULL

    if (dimacs) {
      if (imitsat_connect (&solver->imc,
                           solver->imopts.imitsat_host,
                           solver->imopts.imitsat_port) == 0) {
        if (imitsat_send_hello (&solver->imc, dimacs) != 0) {
          // Fail closed — disable guidance but still solve
          imitsat_close (&solver->imc);
          solver->imopts.imitsat_enabled = 0;
        }
      } else {
        solver->imopts.imitsat_enabled = 0;
      }
      free (dimacs);
    } else {
      // Could not read path (stdin?), do not block solving
      solver->imopts.imitsat_enabled = 0;
    }
  }
  /* ---------------------------------------------------------------------- */

  res = kissat_application (solver, argc, argv);

  // Close ImitSAT socket if open
  if (solver->imc.fd >= 0)
    imitsat_close (&solver->imc);

  kissat_reset_signal_handler ();
  ignore_alarm = true;
  kissat_reset_alarm ();
  kissat_release (solver);
#ifndef NDEBUG
  if (!res)
    return kissat_dump (0);
#endif
  return res;
}
