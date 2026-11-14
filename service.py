#!/usr/bin/env python3
"""Thin HTTP service around a resident SparkSession (Spark local mode).

Endpoints:
  GET  /health  -> {"ok": true, "sparkVersion": "...", "state": "...", "epoch": n}
  POST /query   -> header X-Auth-Token required; body {"sql": "...", "maxRows": 1000, "session": "...", "execution": "...", "config": {...}}
                   returns {"columns": [...], "rows": [[...]], "rowCount": n, "truncated": bool, "epoch": n}
  POST /cancel  -> header X-Auth-Token required; body {"session": "...", "execution": "..."}
                   cancels the running job(s) of that session's job group, or of
                   that execution's job group when "execution" is given;
                   returns {"ok": true} (also when nothing is running)
  POST /admin/config/apply -> header X-Auth-Token required;
                   body {"entries": {"spark.driver.memory": "4g", ...},
                         "jars": [{"type": "url", "value": "https://...", "sha256": "..."},
                                  {"type": "file", "value": "my.jar"}]}
                   applies a new startup-level instance configuration: URL jars
                   are downloaded into the jars directory, the desired config is
                   persisted, running queries are drained (never killed), then
                   the Spark context is rebuilt in-process with the merged
                   config. Returns 202 immediately; poll the status endpoint.
  GET  /admin/config/status -> {"state": "running|draining|restarting|failed_fallback",
                   "epoch": n, "inflight": n,
                   "effective": {"entries": {...}, "jars": ["name.jar", ...]},
                   "lastError": "..." | null}
  GET  /admin/jars -> {"jars": [{"name": "...", "sizeBytes": n}, ...]}
                   lists the files in the server-side jars directory.

The optional "session" field (string, <=128 chars) selects a per-caller Spark
session derived from the resident session via newSession(): current database,
temporary views and session-level config are isolated per session, while
metastore databases and tables stay shared. Requests without "session" use
the default (resident) session, exactly as before this field existed.
Same-session queries are serialized with a per-session lock; different
sessions run in parallel. At most 32 named sessions are kept; sessions idle
for 6 hours are recycled, least recently used first when over the cap.
Session state lives in memory only: a service restart clears it (metastore
data is unaffected). Single statement per request. Values are serialized
JSON-safe (everything else -> str).

The optional "execution" field (string, <=128 chars) scopes a query to its
own job group "session:execution" (default session: "default:execution").
Without it a query runs in the session-level group, exactly as before this
field existed, and /cancel without "execution" still cancels the whole
session group. Cancelling one execution therefore never kills a later
execution in the same session, even when the cancel arrives late.

The optional "config" field on /query (object of string->string, named
sessions only) carries runtime-level Spark config for that session: entries
are applied to the derived session (conf.set) before the statement runs.
Only keys Spark itself reports as modifiable at runtime (SQLConf) are
accepted; a startup-level key is rejected with 400
{"code": "startup_key_in_profile", "key": ...} and an unknown/malformed key
with 400 {"code": "invalid_key", "key": ...}. Startup-level configuration
belongs to the instance config applied through /admin/config/apply.

An instance-config rebuild bumps the epoch and invalidates every named
session created under the old epoch: the next /query naming such a session
gets 409 {"code": "context_rebuilt", "epoch": n} once, after which the name
can be connected afresh. While a rebuild drains or restarts, /query gets
503 {"code": "context_draining"|"context_restarting", "inflight": n};
already running queries always finish naturally (use /cancel to hurry a
drain). If the new config cannot start Spark, the service falls back to the
last known-good config and reports state "failed_fallback" with lastError.

Configuration (environment variables):
  SPARK_API_TOKEN    Required. Requests without a matching X-Auth-Token
                     header are rejected; when unset, every /query is 401.
  PORT               HTTP listen port (default 8080).
  SPARK_DATA_DIR     Metastore + warehouse directory (default /data/spark).
                     Mount a volume here to keep databases and tables.
                     The instance config lives in <dir>/config/, downloaded
                     and operator-placed jars in <dir>/jars/.
  SPARK_MASTER       Spark master (default local[*]).
  SPARK_DRIVER_MEMORY  Driver memory (default 2g). Base value; an applied
                     instance config may override it per restart.
  SPARK_APP_NAME     Spark application name (default spark-sql-service).
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from decimal import Decimal
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import sys
import glob

sys.path += ["/opt/spark/python"] + glob.glob("/opt/spark/python/lib/py4j-*-src.zip")

TOKEN = os.environ.get("SPARK_API_TOKEN", "")
MAX_ROWS_HARD = 5000
MAX_SESSIONS = 32
SESSION_IDLE_SECONDS = 6 * 3600
SESSION_KEY_MAX = 128

DATA_DIR = os.environ.get("SPARK_DATA_DIR", "/data/spark")
MASTER = os.environ.get("SPARK_MASTER", "local[*]")
DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "2g")
APP_NAME = os.environ.get("SPARK_APP_NAME", "spark-sql-service")

CONFIG_DIR = DATA_DIR + "/config"
JARS_DIR = DATA_DIR + "/jars"
DESIRED_PATH = CONFIG_DIR + "/desired.json"
LAST_GOOD_PATH = CONFIG_DIR + "/last-good.json"

MAX_CONFIG_ENTRIES = 50
CONFIG_KEY_MAX = 128
CONFIG_VALUE_MAX = 512
MAX_JARS = 20
JAR_DOWNLOAD_TIMEOUT = 120
JAR_MAX_BYTES = 200 * 1024 * 1024
REBUILD_TIMEOUT_SECONDS = 120

_KEY_RE = re.compile(r"^spark\.[A-Za-z0-9_.]+$")
_MEMORY_VALUE_RE = re.compile(r"^[0-9]+[kKmMgGtT]?$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Keys the service itself owns; an instance config may not move them.
MANAGED_KEYS = {
    "spark.master",
    "spark.jars",
    "spark.sql.warehouse.dir",
    "spark.local.dir",
    "spark.driver.extraJavaOptions",
    "spark.ui.enabled",
}

# spark.sql.* keys that are static despite the prefix (SQLConf static entries
# and catalog wiring). Only used as a fallback when the live isModifiable
# check is unavailable; the live check is authoritative.
STATIC_SQL_KEYS = {"spark.sql.extensions", "spark.sql.warehouse.dir"}

from pyspark.sql import SparkSession


def _fresh_builder():
    """PySpark's SparkSession.builder is a process-wide singleton whose
    options accumulate across builds; clear them so a rebuild only ever
    sees the config meant for it (removing an entry must take effect)."""
    builder = SparkSession.builder
    opts = getattr(builder, "_options", None)
    if isinstance(opts, dict):
        opts.clear()
    return builder


def _clear_active_context():
    """Best-effort cleanup after a stop or a failed getOrCreate(): shut the
    py4j gateway down and clear PySpark's class-level refs, so the next
    build launches a FRESH JVM. Verified on Spark 3.5.3 in this deployment:
    SparkContext.stop() alone leaves the gateway JVM alive and the next
    context silently reuses it — same pid, same heap, so an applied
    spark.driver.memory would never take effect. After a gateway shutdown
    the next build gets a new JVM whose max heap equals the new config.
    Clearing the refs is what makes a later build safe; actually ending
    the old JVM process is a separate step (_ensure_jvm_dead), because a
    killed JVM whose gateway is still referenced breaks every later
    build, while a merely shut-down gateway leaves the JVM — and its
    booted Derby metastore — alive forever."""
    try:
        from pyspark import SparkContext

        active = getattr(SparkContext, "_active_spark_context", None)
        if active is not None:
            try:
                active.stop()
            except Exception:
                pass
            SparkContext._active_spark_context = None
        gw = getattr(SparkContext, "_gateway", None)
        if gw is not None:
            try:
                gw.shutdown()
            except Exception:
                pass
        SparkContext._gateway = None
        SparkContext._jvm = None
    except Exception:
        pass


def _jvm_pid(session):
    try:
        name = session.sparkContext._jvm.java.lang.management.ManagementFactory.getRuntimeMXBean().getName()
        return int(str(name).split("@")[0])
    except Exception:
        return None


def _pid_alive(pid):
    if not pid:
        return False
    # The gateway JVM is this process's child: once it dies it stays a
    # zombie (kill(pid, 0) keeps succeeding) until reaped, so try the reap
    # first — a successful reap is the definitive death certificate.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_is_ours(pid):
    """Only ever signal a JVM this process launched (child process check)."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return int(parts[1]) == os.getpid()
    except Exception:
        return False


def _ensure_jvm_dead(pid):
    """Make sure the old gateway JVM process is gone before any new JVM
    boots the embedded Derby metastore. Verified on Spark 3.5.3 in this
    deployment: SparkContext.stop() + gateway.shutdown() do NOT terminate
    the JVM (non-daemon Spark threads keep it alive indefinitely), and it
    keeps Derby booted the whole time — a new JVM that boots first fails
    with Derby XSDB6 and keeps a broken catalog for its whole life. The
    JVM is this process's own child, launched for the context that has
    already been stopped, so after a passive grace period it is
    terminated: SIGTERM, then SIGKILL. Termination is only safe together
    with the gateway-ref clearing in _clear_active_context — a killed JVM
    whose gateway Python still references breaks every later build."""
    if not _pid_alive(pid):
        return
    for _ in range(50):
        if not _pid_alive(pid):
            return
        time.sleep(0.2)
    if not _pid_is_ours(pid):
        print(f"rebuild: jvm pid {pid} still alive but not our child; leaving it", flush=True)
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return
    for _ in range(25):
        if not _pid_alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    for _ in range(50):
        if not _pid_alive(pid):
            return
        time.sleep(0.2)


def _gateway_jvm_pid():
    """Pid of the JVM behind PySpark's current class-level gateway, if any."""
    try:
        from pyspark import SparkContext

        jvm = getattr(SparkContext, "_jvm", None)
        if jvm is None:
            return None
        name = jvm.java.lang.management.ManagementFactory.getRuntimeMXBean().getName()
        return int(str(name).split("@")[0])
    except Exception:
        return None


def _build_spark(entries=None, jar_names=None):
    """Build the resident SparkSession. entries (startup-level config from an
    applied instance config) override the env base values; service-managed
    keys are applied last so an instance config cannot move the data layout.
    jar_names are file names inside JARS_DIR, assembled into spark.jars."""
    builder = (
        _fresh_builder()
        .appName(APP_NAME)
        .master(MASTER)
        .config("spark.driver.memory", DRIVER_MEMORY)
    )
    for k, v in (entries or {}).items():
        if k not in MANAGED_KEYS:
            builder = builder.config(k, v)
    if jar_names:
        builder = builder.config(
            "spark.jars", ",".join(JARS_DIR + "/" + n for n in jar_names)
        )
    builder = (
        builder
        .config("spark.sql.warehouse.dir", DATA_DIR + "/warehouse")
        .config("spark.local.dir", DATA_DIR + "/tmp")
        .config("spark.driver.extraJavaOptions", "-Dderby.system.home=" + DATA_DIR)
        .config("spark.ui.enabled", "false")
    )
    sess = builder.enableHiveSupport().getOrCreate()
    sess.sparkContext.setLogLevel("WARN")
    # Force the SessionState and the Hive metastore client to materialize
    # now: extension/catalog classes from an instance config only fail when
    # they are built, and a Derby metastore still locked by the previous
    # JVM only fails on first catalog access. A config that cannot answer
    # these must count as a failed start so the fallback chain engages
    # instead of reporting success with a silently broken catalog.
    sess.sql("SELECT 1").collect()
    sess.sql("SHOW DATABASES").collect()
    return sess


def _read_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def _write_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _normalize_stored_config(data):
    """A persisted config file -> {"entries": {...}, "jars": [names]} or None."""
    if not data:
        return None
    entries = data.get("entries") or {}
    jars = data.get("jars") or []
    if not isinstance(entries, dict) or not isinstance(jars, list):
        return None
    return {
        "entries": {str(k): str(v) for k, v in entries.items()},
        "jars": [str(n) for n in jars],
    }


# --- Runtime state ------------------------------------------------------------
# state: running | draining | restarting | failed_fallback
# epoch: generation of the running Spark context; bumped on every rebuild.
# effective: the startup config the running context was actually built with.

_STATE_LOCK = threading.Lock()
_STATE = {
    "state": "running",
    "epoch": 1,
    "last_error": None,
    "applying": False,
    "effective": {"entries": {}, "jars": []},
}

_INFLIGHT = 0
_INFLIGHT_LOCK = threading.Lock()


def _inflight_add(delta):
    global _INFLIGHT
    with _INFLIGHT_LOCK:
        _INFLIGHT += delta
        return _INFLIGHT


def _inflight_now():
    with _INFLIGHT_LOCK:
        return _INFLIGHT


def _try_begin_query():
    """Atomic with the drain decision in the apply worker: if a rebuild is
    draining or restarting, return that state and the caller rejects the
    query; otherwise count the query as in-flight (the drain will wait for
    it, however long it runs) and return None."""
    global _INFLIGHT
    with _STATE_LOCK:
        st = _STATE["state"]
        if st in ("draining", "restarting"):
            return st
        with _INFLIGHT_LOCK:
            _INFLIGHT += 1
        return None


def _state_snapshot():
    with _STATE_LOCK:
        return {
            "state": _STATE["state"],
            "epoch": _STATE["epoch"],
            "lastError": _STATE["last_error"],
            "effective": {
                "entries": dict(_STATE["effective"]["entries"]),
                "jars": list(_STATE["effective"]["jars"]),
            },
            "inflight": _inflight_now(),
        }


def _state_update(**kw):
    with _STATE_LOCK:
        _STATE.update(kw)


def _boot_spark():
    """Boot sequence: desired config -> last known-good -> env base. A config
    that boots becomes the new last known-good. Returns the session; raises
    only when even the env base cannot start (genuinely broken install)."""
    desired = _normalize_stored_config(_read_json_file(DESIRED_PATH))
    last_good = _normalize_stored_config(_read_json_file(LAST_GOOD_PATH))
    candidates = []
    if desired:
        candidates.append(("desired", desired))
    if last_good and last_good != desired:
        candidates.append(("last-good", last_good))
    candidates.append(("base", {"entries": {}, "jars": []}))
    boot_error = None
    for label, cfg in candidates:
        try:
            sess = _build_spark(cfg["entries"], cfg["jars"])
        except Exception as e:
            failed_pid = _gateway_jvm_pid()
            _clear_active_context()
            _ensure_jvm_dead(failed_pid)
            boot_error = f"{label} config failed to start: {str(e)[:300]}"
            print(f"boot: {boot_error}", flush=True)
            continue
        _state_update(effective={"entries": dict(cfg["entries"]), "jars": list(cfg["jars"])})
        if label == "desired":
            _write_json_file(LAST_GOOD_PATH, cfg)
        elif boot_error is not None:
            _state_update(state="failed_fallback", last_error=boot_error)
        return sess
    raise RuntimeError(boot_error or "spark failed to start")


spark = _boot_spark()


class _SessionEntry:
    __slots__ = ("session", "lock", "last_used")

    def __init__(self, session):
        self.session = session
        self.lock = threading.Lock()
        self.last_used = time.monotonic()


_DEFAULT_ENTRY = _SessionEntry(spark)
_SESSIONS = {}
_SESSION_EPOCHS = {}
_STALE_SESSIONS = {}
_SESSIONS_LOCK = threading.Lock()


def _drop_entry(key):
    # Never call .stop() on a derived session: sessions from newSession()
    # share the resident SparkContext, and stop() would kill it for everyone.
    # Dropping our reference is enough; Spark/GC reclaim the session state.
    _SESSIONS.pop(key, None)
    _SESSION_EPOCHS.pop(key, None)


def _purge_idle(now):
    stale = [
        k
        for k, e in _SESSIONS.items()
        if now - e.last_used > SESSION_IDLE_SECONDS and not e.lock.locked()
    ]
    for k in stale:
        _drop_entry(k)


def _evict_lru():
    candidates = sorted(
        (e.last_used, k) for k, e in _SESSIONS.items() if not e.lock.locked()
    )
    if candidates:
        _drop_entry(candidates[0][1])
        return True
    return False


def get_session_entry(key):
    """key=None returns the default (resident) session entry."""
    if key is None:
        return _DEFAULT_ENTRY
    now = time.monotonic()
    with _SESSIONS_LOCK:
        _purge_idle(now)
        entry = _SESSIONS.get(key)
        if entry is None:
            while len(_SESSIONS) >= MAX_SESSIONS:
                if not _evict_lru():
                    break
            entry = _SessionEntry(spark.newSession())
            _SESSIONS[key] = entry
            _SESSION_EPOCHS[key] = _state_snapshot()["epoch"]
        entry.last_used = now
        return entry


def jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        import base64
        return base64.b64encode(bytes(v)).decode()
    return str(v)


def _job_group_id(session_key, execution=None):
    """Job group for a session: the session key itself, or "default" for the
    default (resident) session. With an execution id the group is
    "session:execution", so POST /cancel for that execution kills only that
    execution's jobs and a late cancel can never kill a later execution in
    the same session. POST /cancel kills this group."""
    base = session_key if session_key else "default"
    if execution is None:
        return base
    return f"{base}:{execution}"


def _parse_session(payload):
    """Session key from a /query or /cancel body: None/blank means the default
    session. Raises ValueError on a malformed value (caller maps it to 400)."""
    raw_session = payload.get("session")
    if raw_session is None:
        return None
    if not isinstance(raw_session, str) or len(raw_session) > SESSION_KEY_MAX:
        raise ValueError("bad session")
    if raw_session.strip() == "":
        return None
    return raw_session


def _parse_execution(payload):
    """Execution id from a /query or /cancel body: None/blank means no
    execution scoping (session-level job group, as before). Raises
    ValueError on a malformed value (caller maps it to 400)."""
    raw = payload.get("execution")
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw) > SESSION_KEY_MAX:
        raise ValueError("bad execution")
    if raw.strip() == "":
        return None
    return raw


# --- Config partitioning ------------------------------------------------------

def _is_modifiable(key):
    """True when Spark accepts the key at runtime (SQLConf). Live check via
    RuntimeConfig.isModifiable, JVM bridge as fallback, static heuristic as
    last resort."""
    try:
        return bool(spark.conf.isModifiable(key))
    except Exception:
        pass
    try:
        return bool(spark._jsparkSession.conf().isModifiable(key))
    except Exception:
        pass
    if key.startswith("spark.sql.catalog."):
        return False
    if key in STATIC_SQL_KEYS:
        return False
    return key.startswith("spark.sql.")


def _classify_key(key):
    """"runtime" | "startup" | "invalid" for one config key."""
    if not isinstance(key, str) or len(key) > CONFIG_KEY_MAX or not _KEY_RE.match(key):
        return "invalid"
    return "runtime" if _is_modifiable(key) else "startup"


def _parse_profile_config(payload):
    """The optional /query "config" object -> dict of runtime entries.
    Raises ValueError with a (code, key) pair encoded in the message as
    "code:key" for the caller to map to a 400 response."""
    raw = payload.get("config")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("invalid_key:")
    if len(raw) > MAX_CONFIG_ENTRIES:
        raise ValueError("too_many_entries:")
    out = {}
    for k, v in raw.items():
        if not isinstance(v, str) or len(v) > CONFIG_VALUE_MAX:
            raise ValueError(f"invalid_key:{k}")
        cls = _classify_key(k)
        if cls == "invalid":
            raise ValueError(f"invalid_key:{k}")
        if cls == "startup":
            raise ValueError(f"startup_key_in_profile:{k}")
        out[k] = v
    return out


def _code_only(sql):
    """Return sql with comments and quoted literals blanked out to spaces, so
    separator checks only ever see real code. Skipped regions: line comments
    (-- to end of line), block comments (/* ... */), single-quoted strings,
    double-quoted strings/identifiers and backtick identifiers. A doubled
    quote ('' / "" / ``) is an escaped quote and stays inside the literal;
    a backslash is an ordinary character (Spark escapedStringLiterals is off
    by default). An unterminated literal swallows the rest, which is safe:
    such SQL fails in Spark on its own."""
    out = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            i = n if j < 0 else j
            out.append(" ")
        elif ch == "/" and nxt == "*":
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        elif ch in "'\"`":
            i += 1
            while i < n:
                if sql[i] == ch:
                    if i + 1 < n and sql[i + 1] == ch:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _has_multiple_statements(sql):
    """True if a code-region ';' is followed by more code. A trailing ';'
    (followed only by whitespace, comments or further empty separators)
    stays tolerated, as the old rstrip-based check did."""
    code = _code_only(sql)
    semi = code.find(";")
    while semi >= 0:
        if code[semi + 1:].strip(" \t\r\n;"):
            return True
        semi = code.find(";", semi + 1)
    return False


def cancel_session_jobs(session_key, execution=None):
    """Cancel the running job(s) of one session's job group (or one
    execution's job group when execution is given), without touching the
    session itself: the SparkSession, its temporary views and its current
    database survive; no job running is a no-op. Runs from the /cancel
    request thread while the query runs in its own thread."""
    spark.sparkContext.cancelJobGroup(_job_group_id(session_key, execution))


def run_query(session, sql, max_rows, job_group=None):
    # The job group lets POST /cancel kill this query from another thread:
    # setJobGroup is thread-local, so it only affects jobs this thread
    # submits; interruptOnCancel makes running tasks stop promptly instead of
    # waiting for the next task boundary. The group is cleared afterwards so
    # a later query on a reused thread never inherits it.
    sc = session.sparkContext
    if job_group:
        sc.setJobGroup(job_group, "zeta query", interruptOnCancel=True)
    try:
        df = session.sql(sql)
        cols = list(df.columns)
        limit = max(1, min(int(max_rows or 1000), MAX_ROWS_HARD))
        rows = df.limit(limit + 1).collect()
        truncated = len(rows) > limit
        rows = rows[:limit]
        out = [[jsonable(v) for v in r] for r in rows]
        return {"columns": cols, "rows": out, "rowCount": len(out), "truncated": truncated}
    finally:
        if job_group:
            # PySpark's SparkContext has no clearJobGroup(); clearing the
            # three local properties is exactly what Scala's clearJobGroup
            # does (None becomes Java null, which removes the property).
            for _key in (
                "spark.jobGroup.id",
                "spark.job.description",
                "spark.job.interruptOnCancel",
            ):
                sc.setLocalProperty(_key, None)


# --- Self-healing -----------------------------------------------------------
# A dead gateway cannot be repaired in-process; exiting lets Docker's
# restart policy relaunch the container with a fresh JVM. Metastore and
# warehouse data live on the data volume and are unaffected.

_EXIT_LOCK = threading.Lock()
_EXITING = False

_GATEWAY_DEAD_MARKERS = (
    "Connection refused",
    "Connection reset",
    "connect to the Java server",
    "stopped SparkContext",
    "Cannot call methods on a stopped",
)


def _is_gateway_dead(exc):
    if type(exc).__name__ in ("Py4JNetworkError", "ConnectionRefusedError"):
        return True
    msg = str(exc)
    return any(m in msg for m in _GATEWAY_DEAD_MARKERS)


def _exit_for_restart(reason):
    global _EXITING
    with _EXIT_LOCK:
        if _EXITING:
            return
        _EXITING = True
    print(
        f"FATAL: spark gateway dead ({reason}); exiting so docker restarts the service",
        flush=True,
    )

    def _go():
        time.sleep(0.5)
        os._exit(70)

    threading.Thread(target=_go, daemon=True).start()


def _probe_gateway():
    """One live round-trip through the default session; raises on failure."""
    _ = spark.version
    spark.sql("SELECT 1").collect()


def _gateway_confirmed_dead():
    # Require two failures in a row, a few seconds apart, so a single
    # transient py4j hiccup never restarts the service.
    for attempt in range(2):
        try:
            _probe_gateway()
            return False
        except Exception as e:
            print(f"watchdog: gateway probe failed ({attempt + 1}/2): {e!r:.200}", flush=True)
            if attempt == 0:
                time.sleep(3)
    return True


def _watchdog():
    while True:
        time.sleep(30)
        # Never probe mid-rebuild: the old context is stopped on purpose and
        # a probe would read that as a dead gateway.
        if _state_snapshot()["state"] in ("draining", "restarting"):
            continue
        try:
            _probe_gateway()
        except Exception:
            # A probe that started just before a rebuild began may fail
            # against the deliberately stopped context; never confirm
            # death or exit while a rebuild owns the lifecycle.
            if _state_snapshot()["state"] in ("draining", "restarting"):
                continue
            if _gateway_confirmed_dead():
                _exit_for_restart("watchdog probe")
                return


# --- Instance config: jars ----------------------------------------------------

def _jar_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_url_jar(url, sha256_expected):
    """Download one URL jar into JARS_DIR (or reuse a matching file already
    there). Returns the local file name; raises ValueError on any failure."""
    name = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if not name.endswith(".jar") or "/" in name or ".." in name:
        raise ValueError(f"jar url does not end in a .jar file name: {url[:120]}")
    os.makedirs(JARS_DIR, exist_ok=True)
    dest = os.path.join(JARS_DIR, name)
    if os.path.exists(dest):
        if not sha256_expected or _jar_file_sha256(dest) == sha256_expected.lower():
            return name
    tmp = dest + ".download"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "zeta-spark-service"})
        with urllib.request.urlopen(req, timeout=JAR_DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > JAR_MAX_BYTES:
                    raise ValueError(f"jar too large (> {JAR_MAX_BYTES} bytes): {name}")
                f.write(chunk)
        if sha256_expected and _jar_file_sha256(tmp) != sha256_expected.lower():
            raise ValueError(f"jar sha256 mismatch: {name}")
        os.replace(tmp, dest)
    except ValueError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise ValueError(f"jar download failed: {name}: {str(e)[:200]}")
    return name


def _list_jars():
    out = []
    try:
        for name in sorted(os.listdir(JARS_DIR)):
            if name.endswith(".jar"):
                try:
                    out.append({"name": name, "sizeBytes": os.path.getsize(os.path.join(JARS_DIR, name))})
                except OSError:
                    pass
    except OSError:
        pass
    return out


# --- Instance config: validation ----------------------------------------------

def _validate_apply(payload):
    """Synchronous validation of an /admin/config/apply body. Returns
    (entries, jar_specs) where jar_specs is a list of dicts; raises
    ValueError with a client-facing message (-> 400)."""
    if not isinstance(payload, dict):
        raise ValueError("body must be an object")
    entries = payload.get("entries") or {}
    jars = payload.get("jars") or []
    if not isinstance(entries, dict) or not isinstance(jars, list):
        raise ValueError("entries must be an object, jars an array")
    if len(entries) > MAX_CONFIG_ENTRIES:
        raise ValueError(f"too many entries (> {MAX_CONFIG_ENTRIES})")
    clean_entries = {}
    for k, v in entries.items():
        if not isinstance(k, str) or len(k) > CONFIG_KEY_MAX or not _KEY_RE.match(k):
            raise ValueError(f"invalid config key: {k}")
        if k in MANAGED_KEYS:
            raise ValueError(f"config key is managed by the service: {k}")
        if not isinstance(v, str) or len(v) > CONFIG_VALUE_MAX:
            raise ValueError(f"invalid value for config key: {k}")
        if k.lower().endswith("memory") and not _MEMORY_VALUE_RE.match(v):
            raise ValueError(f"invalid memory value for {k}: {v}")
        clean_entries[k] = v
    if len(jars) > MAX_JARS:
        raise ValueError(f"too many jars (> {MAX_JARS})")
    clean_jars = []
    seen_names = set()
    for spec in jars:
        if not isinstance(spec, dict):
            raise ValueError("jar entries must be objects")
        typ = spec.get("type")
        val = spec.get("value")
        if not isinstance(val, str) or not val or len(val) > 1024:
            raise ValueError("bad jar value")
        if typ == "url":
            if not (val.startswith("https://") or val.startswith("http://")):
                raise ValueError(f"jar url must start with http(s)://: {val[:80]}")
            name = val.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
            sha = spec.get("sha256")
            if sha is not None and (not isinstance(sha, str) or not _SHA256_RE.match(sha)):
                raise ValueError(f"bad sha256 for jar: {name}")
            clean_jars.append({"type": "url", "value": val, "sha256": sha, "name": name})
        elif typ == "file":
            if "/" in val or ".." in val or not val.endswith(".jar"):
                raise ValueError(f"bad jar file name: {val}")
            if not os.path.exists(os.path.join(JARS_DIR, val)):
                raise ValueError(f"jar not found on server: {val}")
            clean_jars.append({"type": "file", "value": val, "name": val})
        else:
            raise ValueError(f"bad jar type: {typ}")
        if clean_jars[-1]["name"] in seen_names:
            raise ValueError(f"duplicate jar: {clean_jars[-1]['name']}")
        seen_names.add(clean_jars[-1]["name"])
    return clean_entries, clean_jars


# --- Instance config: rebuild machinery ----------------------------------------

def _swap_spark(new_sess, new_effective):
    """Install a freshly built resident session: old derived sessions become
    stale (their next /query gets one 409 context_rebuilt), the default
    entry follows the new session, the epoch bumps."""
    global spark, _DEFAULT_ENTRY
    with _SESSIONS_LOCK:
        for name, ep in list(_SESSION_EPOCHS.items()):
            _STALE_SESSIONS[name] = ep
        _SESSIONS.clear()
        _SESSION_EPOCHS.clear()
        spark = new_sess
        _DEFAULT_ENTRY = _SessionEntry(new_sess)
    with _STATE_LOCK:
        _STATE["epoch"] += 1
        _STATE["effective"] = {
            "entries": dict(new_effective["entries"]),
            "jars": list(new_effective["jars"]),
        }


def _stop_current_spark():
    pid = _jvm_pid(spark)
    try:
        spark.stop()
    except Exception as e:
        print(f"rebuild: spark.stop() raised: {e!r:.200}", flush=True)
    _clear_active_context()
    _ensure_jvm_dead(pid)


def _apply_worker(entries, jar_specs):
    """Background thread for one accepted apply: resolve jars, persist the
    desired config, drain in-flight queries, rebuild, fall back if needed.
    Never raises; every outcome lands in the state snapshot."""
    try:
        jar_names = []
        for spec in jar_specs:
            if spec["type"] == "url":
                jar_names.append(_fetch_url_jar(spec["value"], spec.get("sha256")))
            else:
                if not os.path.exists(os.path.join(JARS_DIR, spec["name"])):
                    raise ValueError(f"jar not found on server: {spec['name']}")
                jar_names.append(spec["name"])
        desired = {"entries": dict(entries), "jars": jar_names}
        snap = _state_snapshot()
        if desired == snap["effective"]:
            _write_json_file(DESIRED_PATH, desired)
            _state_update(state="running", last_error=None)
            return
        _write_json_file(DESIRED_PATH, desired)
        _state_update(state="draining")
        while _inflight_now() > 0:
            time.sleep(0.2)
        _state_update(state="restarting")
        _stop_current_spark()
        try:
            new_sess = _build_spark(desired["entries"], desired["jars"])
        except Exception as e:
            err = f"new config failed to start: {str(e)[:300]}"
            print(f"rebuild: {err}; falling back to last known-good", flush=True)
            last_good = _normalize_stored_config(_read_json_file(LAST_GOOD_PATH)) or {
                "entries": {},
                "jars": [],
            }
            if last_good == desired:
                last_good = {"entries": {}, "jars": []}
            failed_pid = _gateway_jvm_pid()
            _clear_active_context()
            _ensure_jvm_dead(failed_pid)
            try:
                fallback_sess = _build_spark(last_good["entries"], last_good["jars"])
            except Exception as e2:
                _exit_for_restart(f"rebuild and fallback both failed: {str(e2)[:200]}")
                return
            _swap_spark(fallback_sess, last_good)
            _state_update(state="failed_fallback", last_error=err)
            return
        _swap_spark(new_sess, desired)
        _write_json_file(LAST_GOOD_PATH, desired)
        _state_update(state="running", last_error=None)
    except ValueError as e:
        _state_update(state="running", last_error=str(e)[:500])
    except Exception as e:
        _state_update(state="running", last_error=f"apply failed: {str(e)[:400]}")
        print(f"rebuild: apply worker failed: {e!r:.300}", flush=True)
    finally:
        _state_update(applying=False)


# --- HTTP handler ---------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            snap = _state_snapshot()
            if snap["state"] in ("draining", "restarting"):
                # The context is stopped on purpose; the front process is
                # alive and the rebuild is making progress.
                self._send(200, {"ok": True, "state": snap["state"], "epoch": snap["epoch"]})
                return
            try:
                self._send(200, {
                    "ok": True,
                    "sparkVersion": spark.version,
                    "state": snap["state"],
                    "epoch": snap["epoch"],
                })
            except Exception as e:
                if _is_gateway_dead(e):
                    _exit_for_restart("health check")
                try:
                    self._send(503, {"ok": False, "error": str(e)[:200]})
                except Exception:
                    pass
        elif self.path == "/admin/config/status":
            if not self._check_auth():
                self._send(401, {"error": "unauthorized"})
                return
            snap = _state_snapshot()
            self._send(200, {
                "state": snap["state"],
                "epoch": snap["epoch"],
                "inflight": snap["inflight"],
                "effective": snap["effective"],
                "lastError": snap["lastError"],
            })
        elif self.path == "/admin/jars":
            if not self._check_auth():
                self._send(401, {"error": "unauthorized"})
                return
            self._send(200, {"jars": _list_jars()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/query":
            self._handle_query()
        elif self.path == "/cancel":
            self._handle_cancel()
        elif self.path == "/admin/config/apply":
            self._handle_apply()
        else:
            self._send(404, {"error": "not found"})

    def _check_auth(self):
        # Empty TOKEN fails closed for every endpoint, exactly like /query.
        token = self.headers.get("X-Auth-Token", "")
        return bool(TOKEN) and hmac.compare_digest(token, TOKEN)

    def _read_json_body(self, max_bytes=65536):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > max_bytes:
            raise ValueError("bad request size")
        return json.loads(self.rfile.read(length))

    def _handle_query(self):
        if not self._check_auth():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            payload = self._read_json_body()
            sql = (payload.get("sql") or "").strip().rstrip(";").strip()
            if not sql:
                self._send(400, {"error": "empty sql"})
                return
            if _has_multiple_statements(sql):
                self._send(400, {"error": "single statement only"})
                return
            session_key = _parse_session(payload)
            execution_key = _parse_execution(payload)
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        busy = _try_begin_query()
        if busy is not None:
            code = "context_draining" if busy == "draining" else "context_restarting"
            self._send(503, {"code": code, "inflight": _inflight_now()})
            return
        try:
            if session_key is not None:
                with _SESSIONS_LOCK:
                    stale = session_key in _STALE_SESSIONS
                    if stale:
                        del _STALE_SESSIONS[session_key]
                if stale:
                    self._send(409, {"code": "context_rebuilt", "epoch": _state_snapshot()["epoch"]})
                    return
            profile_config = {}
            if "config" in payload and payload.get("config") is not None:
                if session_key is None:
                    self._send(400, {"code": "config_requires_session"})
                    return
                try:
                    profile_config = _parse_profile_config(payload)
                except ValueError as e:
                    code, _, key = str(e).partition(":")
                    body = {"code": code or "invalid_key"}
                    if key:
                        body["key"] = key
                    self._send(400, body)
                    return
            try:
                entry = get_session_entry(session_key)
            except Exception as e:
                if _is_gateway_dead(e):
                    _exit_for_restart("session create")
                self._send(200, {"error": str(e)[:2000]})
                return
            with entry.lock:
                entry.last_used = time.monotonic()
                if profile_config:
                    try:
                        for k, v in profile_config.items():
                            entry.session.conf.set(k, v)
                    except Exception as e:
                        self._send(200, {"error": f"config apply failed: {str(e)[:500]}"})
                        return
                try:
                    result = run_query(
                        entry.session, sql, payload.get("maxRows"),
                        _job_group_id(session_key, execution_key),
                    )
                    result["epoch"] = _state_snapshot()["epoch"]
                    self._send(200, result)
                except Exception as e:
                    if _is_gateway_dead(e):
                        _exit_for_restart("query")
                    self._send(200, {"error": str(e)[:2000]})
                finally:
                    entry.last_used = time.monotonic()
        finally:
            _inflight_add(-1)

    def _handle_cancel(self):
        # Deliberately takes no session lock and creates no session entry:
        # it must run concurrently while the query holds the lock, and
        # cancelling a session with no job running is a 200 no-op. It also
        # stays available while a rebuild drains: cancelling the blocking
        # query is exactly how a drain is hurried along.
        if not self._check_auth():
            self._send(401, {"error": "unauthorized"})
            return
        if _state_snapshot()["state"] == "restarting":
            # The context is stopped on purpose; there is nothing to cancel
            # and touching it would read as a dead gateway.
            self._send(200, {"ok": True})
            return
        try:
            payload = self._read_json_body()
            session_key = _parse_session(payload)
            execution_key = _parse_execution(payload)
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        try:
            cancel_session_jobs(session_key, execution_key)
            self._send(200, {"ok": True})
        except Exception as e:
            if _is_gateway_dead(e):
                _exit_for_restart("cancel")
            self._send(200, {"error": str(e)[:2000]})

    def _handle_apply(self):
        if not self._check_auth():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            payload = self._read_json_body(max_bytes=262144)
            entries, jar_specs = _validate_apply(payload)
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        with _STATE_LOCK:
            if _STATE["applying"] or _STATE["state"] in ("draining", "restarting"):
                busy = True
            else:
                busy = False
                _STATE["applying"] = True
        if busy:
            self._send(409, {"code": "apply_in_progress"})
            return
        threading.Thread(target=_apply_worker, args=(entries, jar_specs), daemon=True).start()
        self._send(202, {"state": _state_snapshot()["state"]})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"spark {spark.version} ready on :{port}", flush=True)
    threading.Thread(target=_watchdog, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
