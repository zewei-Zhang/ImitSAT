#include "imitsat_client.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

static void
imitsat_set_error (imitsat_client *c, const char *msg) {
  if (!c)
    return;
  if (!msg) {
    c->last_error[0] = 0;
    return;
  }
  strncpy (c->last_error, msg, sizeof c->last_error - 1);
  c->last_error[sizeof c->last_error - 1] = 0;
}

static int
set_timeout (int fd, int ms) {
  if (ms <= 0)
    return 0;

  struct timeval tv;
  tv.tv_sec  = ms / 1000;
  tv.tv_usec = (ms % 1000) * 1000;

  if (setsockopt (fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) < 0)
    return -1;
  if (setsockopt (fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv) < 0)
    return -1;

  return 0;
}

static int
write_all (int fd, const void *ptr, size_t n) {
  const char *p = (const char *) ptr;
  while (n) {
    ssize_t k = write (fd, p, n);
    if (k <= 0)
      return -1;
    p += (size_t) k;
    n -= (size_t) k;
  }
  return 0;
}

static int
read_line (int fd, char *buf, size_t cap) {
  size_t n = 0;
  if (!buf || cap == 0)
    return -1;

  while (n + 1 < cap) {
    char ch;
    ssize_t r = read (fd, &ch, 1);
    if (r == 0)
      break;
    if (r < 0)
      return -1;
    if (ch == '\n')
      break;
    buf[n++] = ch;
  }
  buf[n] = 0;
  return (int) n;
}

/* Escape a string for JSON (quotes, backslashes, control chars). */
static char *
json_escape (const char *src) {
  if (!src)
    src = "";

  const unsigned char *p = (const unsigned char *) src;
  size_t len = strlen (src);

  /* Worst case: every byte becomes "\u00XX" (6 chars) plus NUL. */
  size_t cap = len * 6 + 1;
  char *out = (char *) malloc (cap);
  if (!out)
    return 0;

  char *q = out;
  const char *hex = "0123456789abcdef";

  while (*p) {
    unsigned char ch = *p++;
    switch (ch) {
    case '\\':
      *q++ = '\\';
      *q++ = '\\';
      break;
    case '"':
      *q++ = '\\';
      *q++ = '"';
      break;
    case '\n':
      *q++ = '\\';
      *q++ = 'n';
      break;
    case '\r':
      *q++ = '\\';
      *q++ = 'r';
      break;
    case '\t':
      *q++ = '\\';
      *q++ = 't';
      break;
    default:
      if (ch < 0x20) {
        *q++ = '\\';
        *q++ = 'u';
        *q++ = '0';
        *q++ = '0';
        *q++ = hex[ch >> 4];
        *q++ = hex[ch & 15];
      } else {
        *q++ = (char) ch;
      }
      break;
    }
  }
  *q = 0;
  return out;
}

/* -------------------------------------------------------------------------- */
/* Socket lifetime                                                            */
/* -------------------------------------------------------------------------- */

int
imitsat_connect (imitsat_client *c, const char *host, int port) {
  if (!c || !host || port <= 0)
    return -1;

  imitsat_set_error (c, NULL);
  c->fd = -1;

  int fd = socket (AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    imitsat_set_error (c, "socket() failed");
    return -1;
  }

  struct sockaddr_in sa;
  memset (&sa, 0, sizeof sa);
  sa.sin_family = AF_INET;
  sa.sin_port   = htons ((unsigned short) port);

  if (inet_pton (AF_INET, host, &sa.sin_addr) != 1) {
    imitsat_set_error (c, "inet_pton failed");
    close (fd);
    return -1;
  }

  if (connect (fd, (struct sockaddr *) &sa, sizeof sa)) {
    imitsat_set_error (c, strerror (errno));
    close (fd);
    return -1;
  }

  c->fd = fd;
  if (c->timeout_ms > 0)
    (void) set_timeout (c->fd, c->timeout_ms);

  imitsat_dyn_reset (c);
  return 0;
}

void
imitsat_close (imitsat_client *c) {
  if (!c)
    return;
  if (c->fd >= 0) {
    close (c->fd);
    c->fd = -1;
  }
}

/* -------------------------------------------------------------------------- */
/* Protocol                                                                   */
/* -------------------------------------------------------------------------- */

/* {"type":"hello","cnf_dimacs":"..."}\n   →   {"type":"ok",...} */
int
imitsat_send_hello (imitsat_client *c, const char *cnf_dimacs) {
  if (!c || c->fd < 0)
    return -1;

  char *esc = json_escape (cnf_dimacs ? cnf_dimacs : "");
  if (!esc)
    return -1;

  int rc = 0;

  if (write_all (c->fd,
                 "{\"type\":\"hello\",\"cnf_dimacs\":\"",
                 sizeof "{\"type\":\"hello\",\"cnf_dimacs\":\"" - 1) ||
      write_all (c->fd, esc, strlen (esc)) ||
      write_all (c->fd, "\"}\n", sizeof "\"}\n" - 1))
    rc = -1;

  free (esc);
  if (rc)
    return -1;

  char line[256];
  int n = read_line (c->fd, line, sizeof line);
  if (n <= 0)
    return -1;

  return strstr (line, "\"type\":\"ok\"") ? 0 : -1;
}

int
imitsat_next_decision (imitsat_client *c, const char *dyn_text, int *out_lit) {
  if (!c || c->fd < 0 || !out_lit)
    return -1;

  char *esc = json_escape (dyn_text ? dyn_text : " D");
  if (!esc)
    return -1;

  int rc = 0;

  if (write_all (c->fd,
                 "{\"type\":\"decide\",\"dyn_text\":\"",
                 sizeof "{\"type\":\"decide\",\"dyn_text\":\"" - 1) ||
      write_all (c->fd, esc, strlen (esc)) ||
      write_all (c->fd, "\"}\n", sizeof "\"}\n" - 1))
    rc = -1;

  free (esc);
  if (rc)
    return -1;

  char line[256];
  int n = read_line (c->fd, line, sizeof line);
  if (n <= 0)
    return -1;

  int lit = 0;
  const char *p = strstr (line, "\"lit\"");
  if (p) {
    p = strchr (p, ':');
    if (p)
      lit = (int) strtol (p + 1, NULL, 10);
  }

  *out_lit = lit;  /* 0 means “decline” */
  return 0;
}

/* -------------------------------------------------------------------------- */
/* Dyn helpers: decisions only                                                */
/* -------------------------------------------------------------------------- */

void
imitsat_dyn_reset (imitsat_client *c) {
  if (!c)
    return;

  c->num_decisions = 0;
  if (IMITSAT_DYN_CAP > 0)
    c->dyn_buf[0] = 0;
}

/* Record a new accepted decision (signed DIMACS variable). */
void
imitsat_dyn_note_decision (imitsat_client *c, int ext_lit) {
  if (!c)
    return;
  if (c->num_decisions >= IMITSAT_MAX_DECISIONS)
    return;

  c->decisions[c->num_decisions++] = ext_lit;

  /* Mark buffer dirty (rebuilt lazily on next imitsat_dyn_cstr). */
  if (IMITSAT_DYN_CAP > 0)
    c->dyn_buf[0] = 0;
}

/* Build and return " D" or " D d1 D d2 ... D" in c->dyn_buf. */
const char *
imitsat_dyn_cstr (imitsat_client *c) {
  static const char fallback[] = " D";

  if (!c)
    return fallback;

  if (c->num_decisions == 0) {
    /* Just " D". */
    if (IMITSAT_DYN_CAP >= 3) {
      c->dyn_buf[0] = ' ';
      c->dyn_buf[1] = 'D';
      c->dyn_buf[2] = 0;
      return c->dyn_buf;
    }
    if (IMITSAT_DYN_CAP > 0)
      c->dyn_buf[0] = 0;
    return fallback;
  }

  char  *buf = c->dyn_buf;
  size_t cap = IMITSAT_DYN_CAP;

  if (!buf || cap == 0)
    return fallback;

  size_t n = 0;

  /* Start with leading space and first "D <lit>" */
  buf[n++] = ' ';

  int written = snprintf (buf + n, cap - n, "D %d", c->decisions[0]);
  if (written <= 0 || (size_t) written >= cap - n) {
    buf[0] = 0;
    return fallback;
  }
  n += (size_t) written;

  /* Subsequent decisions as " D <lit>" */
  for (unsigned i = 1; i < c->num_decisions && n < cap - 1; ++i) {
    written = snprintf (buf + n, cap - n, " D %d", c->decisions[i]);
    if (written <= 0 || (size_t) written >= cap - n)
      break;
    n += (size_t) written;
  }

  /* Trailing " D" */
  if (n + 3 <= cap) {
    buf[n++] = ' ';
    buf[n++] = 'D';
    buf[n]   = 0;
  } else {
    buf[cap - 1] = 0;
  }

  return buf;
}
