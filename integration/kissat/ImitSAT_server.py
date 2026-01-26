# *************************************************************************
# Copyright (c) 2025 Zewei Zhang
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full license text.
# *************************************************************************
"""
ImitSAT lightweight TCP server for Kissat guidance.

Protocol (line-delimited JSON; one JSON object per line):
  1) Client -> {"type":"hello","cnf_dimacs":"p cnf ...\n..."}
     Server <- {"type":"ok"}
  2) Client -> {"type":"decide","dyn_text":" D ..."}
     Server <- {"type":"decide","lit":<int>,"var":<int>,"sign":<+1|-1>}
  3) Client -> {"type":"bye"}  (or "quit"/"close")
     Server <- {"type":"bye"}
"""
import argparse
import json
import socket
import threading
from datetime import datetime, timezone

from pysat.formula import CNF
from integration.cadical.ImitSAT_CaDiCaL import ImitSATRunner


def _stamp() -> str:
    return datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]


def _log(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def _send_json_line(conn: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(data)


def _recv_json_line(conn: socket.socket, max_bytes: int = 8 * 1024 * 1024):
    """
    Read one '\n'-terminated JSON object. Returns dict or None (EOF).
    If a JSON error occurs, returns {"type":"_parse_error","error": "..."}.
    """
    buf = bytearray()
    while True:
        ch = conn.recv(1)
        if not ch:
            return None
        if ch == b"\n":
            break
        buf.extend(ch)
        if len(buf) > max_bytes:
            return {"type": "_parse_error", "error": "line too long"}
    if not buf:
        return None
    try:
        return json.loads(buf.decode("utf-8"))
    except Exception as e:
        return {"type": "_parse_error", "error": str(e)}


def _handle(conn: socket.socket, addr, runner: ImitSATRunner):
    peer = f"{addr[0]}:{addr[1]}"
    _log(f"[conn {peer}] opened")
    prepared = False
    decide_count = 0

    try:
        while True:
            msg = _recv_json_line(conn)
            if msg is None:
                _log(f"[conn {peer}] EOF")
                break

            mtype = msg.get("type", "")

            # ---------------------- HELLO ----------------------
            if mtype == "hello":
                dimacs = msg.get("cnf_dimacs") or msg.get("cnf") or msg.get("instance")
                if not isinstance(dimacs, str) or "p cnf" not in dimacs:
                    _send_json_line(conn, {"error": "bad_cnf",
                                           "detail": "expected DIMACS in 'cnf_dimacs'/'cnf'/'instance'"})
                    continue
                try:
                    cnf = CNF(from_string=dimacs)
                except Exception as e:
                    _send_json_line(conn, {"error": "parse_cnf_failed", "detail": str(e)})
                    continue

                t0 = datetime.now(timezone.utc)
                runner.prepare_cnf(cnf)
                t1 = datetime.now(timezone.utc)
                dt_ms = (t1 - t0).total_seconds() * 1000.0
                _log(f"[conn {peer}] hello: nv={cnf.nv} nc={len(cnf.clauses)} (prepare {dt_ms:.2f} ms)")
                prepared = True
                _send_json_line(conn, {"type": "ok"})

            # ---------------------- DECIDE ---------------------
            elif mtype == "decide":
                if not prepared:
                    _send_json_line(conn, {"error": "not_prepared",
                                           "detail": "send 'hello' with 'cnf_dimacs' first"})
                    continue

                dyn = msg.get("dyn_text") or msg.get("events") or " D"
                t0 = datetime.now(timezone.utc)
                try:
                    var, sign = runner.next_decision_from_dyn_text(dyn)
                except Exception as e:
                    _send_json_line(conn, {"error": "model_error", "detail": str(e)})
                    continue
                t1 = datetime.now(timezone.utc)
                dt_ms = (t1 - t0).total_seconds() * 1000.0

                lit = int(var if sign > 0 else -var) if var else 0
                decide_count += 1
                _log(f"[conn {peer}] decide#{decide_count}: dyn={dyn!r} -> lit={lit} (var={var}, sign={sign}) "
                     f"model {dt_ms:.2f} ms")
                _send_json_line(conn, {"type": "decide", "lit": lit, "var": var, "sign": sign})

            # ----------------------- BYE -----------------------
            elif mtype in ("bye", "quit", "close"):
                _send_json_line(conn, {"type": "bye"})
                _log(f"[conn {peer}] bye (decides={decide_count})")
                break

            # --------------- PARSE / UNKNOWN ERRORS ------------
            elif mtype == "_parse_error":
                err = msg.get("error", "JSON parse error")
                _send_json_line(conn, {"error": "parse_error", "detail": err})
                _log(f"[conn {peer}] parse_error: {err}")

            else:
                _send_json_line(conn, {"error": "unknown_type", "got": mtype})
                _log(f"[conn {peer}] unknown_type: {mtype!r}")

    except Exception as e:
        _log(f"[conn {peer}] exception: {type(e).__name__}: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        _log(f"[conn {peer}] closed")


def main():
    ap = argparse.ArgumentParser(description="ImitSAT TCP server (Kissat guidance).")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    ap.add_argument("--model-dir", type=str, default="./model_ckpt/", help="ImitSAT model directory")
    ap.add_argument("--model-config", type=str, default="./model_config/ImitSAT_config.json",
                    help="ImitSAT model config JSON")
    args = ap.parse_args()

    runner = ImitSATRunner(model_dir=args.model_dir, model_config=args.model_config)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(16)
        _log(f"[ImitSAT server] listening on {args.host}:{args.port}")

        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=_handle, args=(conn, addr, runner), daemon=True)
            t.start()


if __name__ == "__main__":
    main()
