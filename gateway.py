"""Local inference gateway: Claude Desktop -> OpenCode free-tier models.

Exposes Anthropic Messages API (/v1/messages) + OpenAI chat API
(/v1/chat/completions) + models list (/v1/models), and forwards every
request to a local `opencode serve` instance (which is the ONLY path that
can use OpenCode's free tier - direct Zen REST calls return 403
FreeTierError for free models).

Flow per chat request (stateless, one opencode session per request):
  1. POST /session {"title": ...} -> {id: ses_...}
  2. POST /session/{id}/message {"model": {providerID, modelID},
     "system": ..., "parts": [{"type": "text", "text": ...}]} -> {info, parts}
  3. Concatenate parts where type == "text" -> assistant reply.
  4. On error / rate-limit / empty reply, try the next model in combo.json.

Upstream is NON-streaming; for stream=true we re-emit the full reply as
Anthropic/OpenAI SSE events so Claude Desktop still works (documented).

v2.0-beta tool bridge: Claude Desktop Cowork/Code always sends `tools`
(computer-use etc.) plus params like `tool_choice`, `betas`,
`metadata`, `service_tier`, `output_config`. When tools are present the
gateway builds a TOOL-DECISION prompt, asks the upstream combo
(chat_with_fallback) to decide, and converts its STRICT-JSON decision
into Anthropic content blocks (text and/or tool_use). The CLIENT executes
tools; the gateway only DECIDES (never executes actions). Loop state
lives in Claude Desktop; each request carries full history (stateless).
Unknown query params (e.g. ?beta=true) are ignored.

v3.0-beta vision bridge: Anthropic image blocks (base64 PNG/JPEG, top-level
or nested in tool_result content) are decoded, magic-byte verified, capped
(MAX 5 images / 4MB total per request) and saved under sandbox/img/.
When images are present the gateway routes vision-PASS models FIRST
(VISION_RANK order), then falls back to remaining combo models text-only.
Serve attachment format (verified 2026-09-24, opencode 1.18.32):
message part {"type": "file", "mime": "image/png", "url": "file://<abs path>"}.
Per-model image rejection (400/422) is an UpstreamError -> next fallback.
"""

import base64
import json
import logging
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # .env overrides nothing; real env wins


def _int_env(name: str, default: int) -> int:
    try:
        val = int(os.getenv(name, str(default)))
        return val if val > 0 else default
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        val = float(os.getenv(name, str(default)))
        return val if val > 0 else default
    except Exception:
        return default


# ---------------- config (no secrets are ever logged) ----------------
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "local-dev-key-12345")
OPENCODE_SERVE_URL = os.getenv("OPENCODE_SERVE_URL", "http://127.0.0.1:4097").rstrip("/")
OPENCODE_SERVE_USERNAME = os.getenv("OPENCODE_SERVE_USERNAME", "opencode")
OPENCODE_SERVE_PASSWORD = os.getenv("OPENCODE_SERVE_PASSWORD", "")
OPENCODE_EXE = os.getenv(
    # Generic default: expects `opencode` on PATH.
    # Windows npm example: %AppData%\\npm\\node_modules\\opencode-ai\\bin\\opencode.exe
    "OPENCODE_EXE",
    "opencode",
)
PORT = int(os.getenv("PORT", "3457"))
COMBO_PATH = os.getenv("COMBO_PATH", "combo.json")
# Tools lockdown is OFF by default: passing {tool: false} for every tool makes
# several free models return empty replies instantly (diagnosed 2026-09-24).
# With tools enabled the model runs inside SESSION_DIR only - keep that dir
# isolated and never store real secrets in it.
DISABLE_TOOLS = os.getenv("GATEWAY_DISABLE_TOOLS", "false").lower() == "true"
SESSION_DIR = BASE_DIR / "sandbox"
UPSTREAM_TIMEOUT = 150.0  # per-model ceiling: free models answer in
# ~5-35s when healthy; a hang beyond this is treated as failure so the
# request falls back (or 502s) instead of stalling Claude Desktop for
# many minutes per model.
# Failure cooldown: a model that just failed is skipped for this long,
# so one bad model doesn't cost a full timeout on EVERY turn while it
# is down (free-tier flakiness comes in waves).
try:
    FAIL_COOLDOWN_S = int(os.getenv("GATEWAY_FAIL_COOLDOWN_S", "600"))
    if FAIL_COOLDOWN_S <= 0:
        FAIL_COOLDOWN_S = 600
except Exception:
    FAIL_COOLDOWN_S = 600
# Wall-clock ceiling for ONE whole request (all combo models + the forced
# retry). Without this an 8-model combo could burn 8 x UPSTREAM_TIMEOUT
# and Claude Desktop would give up long before we answer.
TOTAL_BUDGET_S = _float_env("GATEWAY_TOTAL_BUDGET_S", 600.0)
_COOLDOWN_LOCK = threading.Lock()
_MODEL_COOLDOWN: dict[str, float] = {}  # model_ref -> unix time of last failure
# Inbound images live in sandbox/img for the duration of a request only.
# This TTL (hours) bounds that dir so long sessions don't fill the disk.
IMG_TTL_H = _float_env("GATEWAY_IMG_TTL_H", 6.0)
GATEWAY_CAPTURE = os.getenv("GATEWAY_CAPTURE", "0") == "1"
IMG_DIR = SESSION_DIR / "img"  # v3: saved inbound images for vision routing

# ---------------- vision config (v3.0-beta) ----------------
# Verified vision matrix, probed 2026-09-24 via opencode serve (opencode
# 1.18.32) with a 400x200 red PNG reading "ABC123":
#   PASS mimo-v2.6-flash-free (27.5s, perfect description)
#   PASS muse-spark-1.2-contributor-free (9.8s, accurate)
#   PASS muse-spark-1.3-contributor-free (11.3s, accurate)
#   PASS space-bunny-free (8.1s, correct gist, minor border/color quibbles)
#   PASS opencode-go/muse-spark-1.3-contributor (9.5s, accurate; probed 2026-09-25)
#   FAIL ling-3.0-flash-fin-free (explicit vision denial, 3.9s)
#   FAIL nemotron-3-ultra-free (explicit vision denial, 3.8s)
#   FAIL nemotron-3.5-lightning-free (vision denial, 32s)
#   UNSUPPORTED mimo-v2.5-free (HTTP 500 on image AND text-only; model broken)
# Rank: PASS models ordered by answer quality, ties broken by latency.
VISION_RANK = [
    # Added 2026-09-25 per user request; probed same day: PASS, ~9.5s,
    # accurate description. First because it is the fastest verified vision
    # model and avoids the client retry loop on slow vision fallbacks.
    "opencode-go/muse-spark-1.3-contributor",
    "opencode/mimo-v2.6-flash-free",
    "opencode/muse-spark-1.2-contributor-free",
    "opencode/muse-spark-1.3-contributor-free",
    "opencode/space-bunny-free",
]
VISION_PROBE_DATE = "2026-09-24"


def _parse_vision_rank_order() -> list[str] | None:
    """Optional env override: comma-separated model refs (bare or provider/id)."""
    raw = os.getenv("VISION_RANK_ORDER", "").strip()
    if not raw:
        return None
    refs = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "/" not in tok:
            tok = "opencode/" + tok
        refs.append(tok)
    return refs or None


VISION_RANK_OVERRIDE = _parse_vision_rank_order()


def vision_rank() -> list[str]:
    """Effective vision routing order (env override wins over hardcoded rank)."""
    return VISION_RANK_OVERRIDE or VISION_RANK


MAX_IMAGES = _int_env("GATEWAY_MAX_IMAGES", 5)
MAX_IMAGE_BYTES = int(_float_env("GATEWAY_MAX_IMAGE_MB", 4.0) * 1024 * 1024)

_PNG_MAGIC = b"\x89PNG"
_JPEG_MAGIC = b"\xff\xd8\xff"

STARTED_AT = time.time()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")


def _mask(value: str) -> str:
    """Mask a secret for logs: first 2 + **** + last 2 (never full)."""
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}****{value[-2:]}"


def load_combo() -> dict:
    path = Path(COMBO_PATH)
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("aliases", {})


COMBO = load_combo()  # {alias: [model, ...]}, reloaded on restart


# ---------------- opencode serve client ----------------
def _new_client(**kw) -> httpx.Client:
    """httpx client for the LOCAL serve only.

    trust_env=False is deliberate and important: httpx reads HTTP(S)_PROXY /
    NO_PROXY from the environment, and a NO_PROXY containing a bracketed IPv6
    host (e.g. "::1,[::1]") makes httpx>=0.28 raise InvalidURL while building
    the client - crashing the gateway before any request is sent. We only ever
    talk to 127.0.0.1, so proxies are never wanted.
    """
    return httpx.Client(trust_env=False, **kw)


def _basic_auth() -> str:
    raw = f"{OPENCODE_SERVE_USERNAME}:{OPENCODE_SERVE_PASSWORD}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def serve_headers() -> dict:
    return {"Authorization": _basic_auth(), "Content-Type": "application/json"}


_serve_proc: subprocess.Popen | None = None


def serve_probe(timeout: float = 5.0) -> tuple[bool, str]:
    """Return (reachable_with_valid_auth, detail). No secrets in detail."""
    try:
        with _new_client(timeout=timeout) as c:
            r = c.get(OPENCODE_SERVE_URL + "/config", headers={"Authorization": _basic_auth()})
        if r.status_code == 200:
            return True, "ok"
        if r.status_code == 401:
            return False, "auth-rejected (username/password mismatch)"
        return False, f"http-{r.status_code}"
    except Exception as e:
        return False, f"unreachable ({type(e).__name__})"


def ensure_serve_running() -> bool:
    """Use existing serve if reachable, else spawn `opencode serve` ourselves."""
    global _serve_proc
    ok, detail = serve_probe()
    if ok:
        log.info("opencode serve already reachable (%s)", detail)
        return True
    log.info("serve not usable (%s); spawning own instance", detail)
    parts = urllib.parse.urlparse(OPENCODE_SERVE_URL)
    port = parts.port or 4097
    host = parts.hostname or "127.0.0.1"
    env = dict(os.environ)
    # Serve inherits the user's env (auth.json keys etc.); we only set Basic-auth creds.
    env["OPENCODE_SERVER_USERNAME"] = OPENCODE_SERVE_USERNAME
    env["OPENCODE_SERVER_PASSWORD"] = OPENCODE_SERVE_PASSWORD
    try:
        _serve_proc = subprocess.Popen(
            [OPENCODE_EXE, "serve", "--port", str(port), "--hostname", host],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log.error("failed to spawn opencode serve: %s", type(e).__name__)
        return False
    for _ in range(35):
        time.sleep(1)
        ok, _ = serve_probe()
        if ok:
            log.info("spawned opencode serve is up")
            return True
        if _serve_proc.poll() is not None:
            log.error("spawned serve exited early (code %s)", _serve_proc.poll())
            return False
    log.error("spawned serve did not come up in time")
    return False


# Cache of {"tool_name": False} to disable every server tool for pure chat.
_TOOLS_MAP: dict | None = None
_TOOLS_FETCHED = False


def get_tools_map(client: httpx.Client) -> dict | None:
    """Fetch tool ids once; return {id: False} or None if unknown/disabled."""
    global _TOOLS_MAP, _TOOLS_FETCHED
    if not DISABLE_TOOLS:
        return None
    if _TOOLS_FETCHED:
        return _TOOLS_MAP
    _TOOLS_FETCHED = True
    try:
        r = client.get(OPENCODE_SERVE_URL + "/experimental/tool/ids",
                       headers={"Authorization": _basic_auth()}, timeout=10.0)
        if r.status_code != 200:
            log.info("tool lockdown skipped (tool/ids http-%s)", r.status_code)
            return None
        data = r.json()
        ids = data if isinstance(data, list) else data.get("ids", data.get("tools", []))
        names = []
        for item in ids or []:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                names.append(item["id"])
        if not names:
            log.info("tool lockdown skipped (no tool ids parsed)")
            return None
        _TOOLS_MAP = {name: False for name in names}
        log.info("tool lockdown active for %d tools", len(names))
    except Exception as e:
        log.info("tool lockdown skipped (%s)", type(e).__name__)
        return None
    return _TOOLS_MAP


class UpstreamError(Exception):
    pass


# Longest useful transcript before we insist on a long-context model.
# 150k chars ~ 40k tokens: beyond this the small free models start degrading.
BIG_PROMPT_CHARS = _int_env("GATEWAY_BIG_PROMPT_CHARS", 150000)

# Real context windows, read from the opencode serve /provider endpoint on
# 2026-09-25 (free models cost 0; opencode-go is a PAID tier at $0.1/Mtok in).
MODEL_CONTEXT: dict[str, int] = {
    "opencode/mimo-v2.6-flash-free": 200_000,
    "opencode/mimo-v2.5-free": 200_000,
    "opencode/ling-3.0-flash-fin-free": 262_144,
    "opencode/nemotron-3-ultra-free": 1_000_000,
    "opencode/nemotron-3.5-lightning-free": 1_000_000,
    "opencode/space-bunny-free": 1_048_576,
    "opencode/muse-spark-1.2-contributor-free": 1_048_576,
    "opencode/muse-spark-1.3-contributor-free": 1_048_576,
    "opencode-go/muse-spark-1.3-contributor": 1_048_576,
}


def _model_context(ref: str) -> int:
    """Context window for a model ref; 0 = unknown (never filtered out)."""
    _, model_id = split_model(ref)
    if model_id in MODEL_CONTEXT:
        return MODEL_CONTEXT[model_id]
    for known, ctx in MODEL_CONTEXT.items():
        if known.split("/", 1)[-1] == model_id:
            return ctx
    return 0


def split_model(ref: str) -> tuple[str, str]:
    """'opencode/mimo-v2.6-flash-free' -> ('opencode', 'mimo-v2.6-flash-free')."""
    if "/" in ref:
        provider, model = ref.split("/", 1)
        return provider, model
    return "opencode", ref


def resolve_session_dir(project_paths: list[str] | None) -> Path:
    """Where the model actually RUNS for this request.

    This is the root fix for "the model says it entered my folder but never
    did". Claude Desktop tells us the opened folder in the system prompt
    ("Working directory: ..."), but the gateway used to hardcode its own
    sandbox as the serve session directory. The model was therefore told to
    read the user's project while every tool call executed inside the sandbox
    - so it "read" sandbox files, reported nonsense, and looped.

    Now the session runs in the folder the user actually opened, so relative
    paths (Read("main.py"), Glob("*")) resolve against their project and the
    model genuinely has read/write access there - like real Claude Code.

    Falls back to the sandbox when the folder is missing or not a directory,
    so a stale/renamed path can never break the request.
    """
    for p in (project_paths or []):
        try:
            cand = Path(p)
            if cand.is_dir():
                return cand
        except Exception:
            continue
    return SESSION_DIR


def chat_via_serve(client: httpx.Client, model_ref: str, system: str, prompt: str,
                   image_paths: list[dict] | None = None,
                   work_dir: Path | None = None) -> tuple[str, dict]:
    """One attempt with a single free model. Returns (text, usage). Raises UpstreamError.

    image_paths: [{"path", "mime"}] attached as serve file parts
    ({"type": "file", "mime": ..., "url": "file://..."}). A 400/422 on an
    image request means the model rejected the format -> UpstreamError so
    chat_with_fallback moves on (never aborts the whole request).

    work_dir: the user's opened project folder, so the model runs THERE.
    Defaults to the isolated sandbox.
    """
    provider_id, model_id = split_model(model_ref)
    session_dir = work_dir or SESSION_DIR
    # 1. fresh stateless session, in the user's project folder when known
    try:
        r = client.post(
            OPENCODE_SERVE_URL + "/session",
            params={"directory": str(session_dir)},
            json={"title": "gateway chat"},
            headers=serve_headers(),
        )
    except Exception as e:
        raise UpstreamError(f"session create failed: {type(e).__name__}") from e
    if r.status_code != 200:
        raise UpstreamError(f"session create http-{r.status_code}: {r.text[:200]}")
    session_id = r.json().get("id")
    if not session_id:
        raise UpstreamError("session create returned no id")

    # 2. send prompt; serve answers synchronously as JSON {info, parts}
    parts: list[dict] = [{"type": "text", "text": prompt}]
    has_images = bool(image_paths)
    if has_images:
        for img in image_paths:
            parts.append({"type": "file", "mime": img.get("mime", "image/png"),
                          "url": _serve_file_url(img["path"])})
    body: dict = {
        "model": {"providerID": provider_id, "modelID": model_id},
        "parts": parts,
    }
    if system:
        body["system"] = system
    tools_map = get_tools_map(client)
    if tools_map:
        body["tools"] = tools_map
    try:
        r = client.post(f"{OPENCODE_SERVE_URL}/session/{session_id}/message",
                        json=body, headers=serve_headers())
    except Exception as e:
        raise UpstreamError(f"prompt failed: {type(e).__name__}") from e
    if r.status_code != 200:
        if has_images and r.status_code in (400, 422):
            raise UpstreamError(f"image parts rejected http-{r.status_code}: {r.text[:300]}")
        raise UpstreamError(f"prompt http-{r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except Exception as e:
        raise UpstreamError("prompt returned non-JSON") from e

    parts = data.get("parts", []) if isinstance(data, dict) else []
    texts = [p.get("text", "") for p in parts
             if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
    text = "".join(texts).strip()
    info = data.get("info", {}) if isinstance(data, dict) else {}
    tokens = info.get("tokens", {}) if isinstance(info, dict) else {}
    usage = {
        "input_tokens": int(tokens.get("input", 0) or 0),
        "output_tokens": int(tokens.get("output", 0) or 0),
    }
    if not text:
        raise UpstreamError(f"empty reply (finish={info.get('finish') if isinstance(info, dict) else '?'})")
    return text, usage


def _prune_cooldown(now: float) -> None:
    """Drop cooldown entries whose window has fully elapsed (bounded dict)."""
    with _COOLDOWN_LOCK:
        stale = [ref for ref, ts in _MODEL_COOLDOWN.items()
                 if now - ts >= FAIL_COOLDOWN_S]
        for ref in stale:
            _MODEL_COOLDOWN.pop(ref, None)


def _cooldown_active(ref: str, now: float) -> bool:
    with _COOLDOWN_LOCK:
        ts = _MODEL_COOLDOWN.get(ref)
    return ts is not None and now - ts < FAIL_COOLDOWN_S


def _cooldown_mark_failed(ref: str, now: float) -> None:
    with _COOLDOWN_LOCK:
        _MODEL_COOLDOWN[ref] = now


def _cooldown_clear(ref: str) -> None:
    with _COOLDOWN_LOCK:
        _MODEL_COOLDOWN.pop(ref, None)


def chat_with_fallback(client: httpx.Client, alias: str, system: str, prompt: str,
                       image_paths: list[dict] | None = None,
                       deadline: float | None = None,
                       work_dir: Path | None = None) -> tuple[str, str, dict]:
    """Try each combo model in order. Returns (text, model_used, usage).

    Vision routing (v3): when image_paths is non-empty, vision-PASS models
    (VISION_RANK filtered to this alias) are tried FIRST with images
    attached, then the remaining combo models as text-only fallback (with an
    omission note appended). Without images the existing order is untouched.

    deadline: absolute time.time() after which no further model is tried
    (the whole-request budget). Prevents a long combo from stacking one
    UPSTREAM_TIMEOUT per model.
    """
    models = COMBO.get(alias, [])
    if not models:
        raise UpstreamError(f"unknown model alias '{alias}'")
    ordered: list[tuple[str, bool]] = []  # (model_ref, attach_images)
    if image_paths:
        rank = [m for m in vision_rank() if m in models]
        rest = [m for m in models if m not in set(rank)]
        ordered = [(m, True) for m in rank] + [(m, False) for m in rest]
        if rank:
            log.info("vision: %d images -> vision-first order %s",
                     len(image_paths), rank)
        else:
            log.info("vision: %d images but no vision model in alias '%s'; text-only",
                     len(image_paths), alias)
    else:
        ordered = [(m, False) for m in models]
    # Volume routing: a long transcript needs a model with a long context.
    # The free mimo model only holds ~200k tokens; sending it a 400k-char
    # prompt (~100k tokens) is fine, but going far past that is not, so big
    # prompts prefer the 1M-context free models and only fall back to the
    # paid opencode-go entry when the user picked it explicitly.
    if len(prompt) >= BIG_PROMPT_CHARS:
        big = [x for x in ordered if _model_context(x[0]) >= 900_000]
        if big and big != ordered:
            log.info("routing: %d-char prompt -> long-context models %s",
                     len(prompt), [m for m, _ in big])
            ordered = big
    # Failure cooldown: skip models that failed recently (they tend to
    # stay down in waves); if everything is cooled, try all anyway.
    now = time.time()
    _prune_cooldown(now)
    fresh = [(ref, attach) for ref, attach in ordered
             if not _cooldown_active(ref, now)]
    fresh_refs = {ref for ref, _ in fresh}
    skipped = [ref for ref, _ in ordered if ref not in fresh_refs]
    if skipped:
        log.info("cooldown: skipping recently-failed %s", skipped)
    if fresh:
        ordered = fresh
    else:
        log.info("cooldown: all models cooled, trying anyway")
    last_err = "no models configured"
    for idx, (ref, attach) in enumerate(ordered):
        # Whole-request budget: never start a model we have no time for.
        if deadline is not None and time.time() >= deadline:
            last_err = f"request time budget ({TOTAL_BUDGET_S:.0f}s) exhausted"
            log.info("fallback: %s (after %d model(s))", last_err, idx)
            break
        try:
            eff_prompt = prompt
            if image_paths and not attach:
                # NOTE: no count here - flatten_messages already emitted a
                # marker for every omitted image, and the omitted total is
                # also passed separately to the decision prompt. Repeating
                # the number made the model double-count the omissions.
                eff_prompt = (prompt + "\n\n[The image(s) referenced above are NOT "
                                       "visible to you in this text-only fallback.]")
            if attach:
                log.info("vision: %d images -> trying %s (vision-rank #%d)",
                         len(image_paths), ref, idx + 1)
            text, usage = chat_via_serve(client, ref, system, eff_prompt,
                                         image_paths if attach else None,
                                         work_dir=work_dir)
            if image_paths:
                log.info("vision: alias %s answered via %s (%d chars, images=%s)",
                         alias, ref, len(text), "attached" if attach else "omitted")
            else:
                log.info("alias %s answered via %s (%d chars)", alias, ref, len(text))
            _cooldown_clear(ref)  # healthy again: clear cooldown
            return text, ref, usage
        except UpstreamError as e:
            last_err = f"{ref}: {e}"
            _cooldown_mark_failed(ref, time.time())  # cool down this model
            log.info("fallback: %s", last_err[:200])
            continue
    raise UpstreamError(f"all {len(ordered)} combo models failed; last: {last_err[:300]}")


# ---------------- Anthropic/OpenAI translation ----------------
def extract_system(system) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(b.get("text", "") for b in system
                         if isinstance(b, dict) and b.get("type") == "text" and b.get("text"))
    return ""


def _tool_result_text(content) -> str:
    """Extract readable text from a tool_result content (str | list | dict)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "output", "result"):
            val = content.get(key)
            if isinstance(val, str) and val:
                return val
        try:
            return json.dumps(content, ensure_ascii=False)[:2000]
        except Exception:
            return str(content)[:2000]
    if isinstance(content, list):
        chunks = []
        for b in content:
            if isinstance(b, str):
                chunks.append(b)
            elif isinstance(b, dict):
                if b.get("type") in ("text", "output") and b.get("text"):
                    chunks.append(str(b["text"]))
                elif isinstance(b.get("text"), str):
                    chunks.append(b["text"])
        return "\n".join(chunks)
    return str(content)


# ---------------- vision image handling (v3.0-beta) ----------------
def _iter_image_blocks(messages: list):
    """Yield image blocks in document order (top-level + tool_result-nested).

    Covers Anthropic {"type": "image", "source": {...}} and OpenAI
    {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}.
    flatten_messages consumes flags in this exact order - keep in sync.
    """
    for m in messages or []:
        content = m.get("content", "") if isinstance(m, dict) else ""
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image":
                yield b
            elif b.get("type") == "image_url":
                yield b
            elif b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    for nb in c:
                        if isinstance(nb, dict) and nb.get("type") in ("image", "image_url"):
                            yield nb


def _decode_image_block(block: dict) -> tuple[bytes | None, str]:
    """Return (raw_bytes_or_None, mime_guess). Only base64 PNG/JPEG accepted."""
    try:
        if block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "") if isinstance(
                block.get("image_url"), dict) else ""
            if not isinstance(url, str) or not url.startswith("data:"):
                return None, ""
            header, _, b64 = url.partition(",")
            mime = header[5:].split(";")[0] if header.startswith("data:") else ""
            return base64.b64decode(b64, validate=True), mime or "image/png"
        src = block.get("source") or {}
        if not isinstance(src, dict) or src.get("type") != "base64":
            return None, ""
        mime = str(src.get("media_type", "") or "")
        data = src.get("data", "")
        if not isinstance(data, str) or not data:
            return None, mime
        return base64.b64decode(data, validate=True), mime
    except Exception:
        return None, ""


def _ext_for(raw: bytes, mime: str) -> str | None:
    """Magic-byte check; returns 'png'/'jpg' or None (invalid)."""
    if raw.startswith(_PNG_MAGIC):
        return "png"
    if raw.startswith(_JPEG_MAGIC):
        return "jpg"
    return None


def cleanup_images(max_age_s: float | None = None) -> int:
    """Delete saved inbound images older than the TTL. Returns files removed.

    sandbox/img accumulates one file per inbound image; without this a long
    Cowork session would grow the dir (and the disk) without bound. Called
    on startup and after each vision request. Safe to call concurrently.
    """
    ttl = IMG_TTL_H * 3600.0 if max_age_s is None else max_age_s
    if ttl <= 0:
        return 0
    removed = 0
    cutoff = time.time() - ttl
    try:
        entries = list(IMG_DIR.glob("*"))
    except Exception:
        return 0
    for p in entries:
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except Exception:
            continue  # file in use / already gone: skip, never fail a request
    if removed:
        log.info("vision: pruned %d image(s) older than %.1fh",
                 removed, ttl / 3600.0)
    return removed


def extract_request_images(messages: list) -> tuple[list[dict], int, int, list[str]]:
    """Decode/save inbound images. Returns (files, omitted, invalid, status).

    files: [{"path", "mime", "filename"}] for serve file-parts.
    omitted: valid images dropped over the count/byte caps.
    invalid: blocks that failed base64/magic checks (skipped gracefully).
    status: per-image-block "attached"|"omitted"|"invalid", aligned with
    _iter_image_blocks order (flatten_messages consumes this).
    """
    files: list[dict] = []
    status: list[str] = []
    omitted = 0
    invalid = 0
    total_bytes = 0
    try:
        IMG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        log.info("vision: img dir unavailable, all images omitted")
        n = sum(1 for _ in _iter_image_blocks(messages))
        return [], n, 0, ["omitted"] * n
    for block in _iter_image_blocks(messages):
        raw, mime = _decode_image_block(block)
        ext = _ext_for(raw, mime) if raw else None
        if not raw or not ext:
            invalid += 1
            status.append("invalid")
            continue
        if len(files) >= MAX_IMAGES or total_bytes + len(raw) > MAX_IMAGE_BYTES:
            omitted += 1
            status.append("omitted")
            continue
        try:
            name = uuid.uuid4().hex + "." + ext
            dest = IMG_DIR / name
            dest.write_bytes(raw)
            total_bytes += len(raw)
            files.append({"path": str(dest.resolve()),
                          "mime": "image/png" if ext == "png" else "image/jpeg",
                          "filename": name})
            status.append("attached")
        except Exception:
            invalid += 1
            status.append("invalid")
    if files or omitted or invalid:
        log.info("vision: %d attached, %d omitted (over slam limit), %d invalid",
                 len(files), omitted, invalid)
    return files, omitted, invalid, status


def _serve_file_url(abs_path: str) -> str:
    # Verified form (2026-09-24): "file://" + forward-slash abs path.
    return "file://" + abs_path.replace("\\", "/")


# ---- per-turn render limits ------------------------------------------------
# These used to be hard-coded at 2000/500 chars, which silently threw away most
# of every tool result and made the model re-read files to recover the content.
# They are raised (still far below any model's real context window) and are
# tunable without a code edit.
TOOL_RESULT_CHAR_LIMIT = _int_env("GATEWAY_TOOL_RESULT_CHARS", 8000)
TOOL_INPUT_CHAR_LIMIT = _int_env("GATEWAY_TOOL_INPUT_CHARS", 1500)
# Kill switch for the "files you already read" ledger, in case a model ever
# misreads it as a constraint.
PROGRESS_LEDGER = os.getenv("GATEWAY_PROGRESS_LEDGER", "true").lower() != "false"
# Log which project folder the gateway resolved, every request. This is the
# first thing to check when the agent opens the wrong folder.
PROJECT_LOG = os.getenv("GATEWAY_PROJECT_LOG", "true").lower() != "false"


def flatten_messages(messages: list,
                     image_status: list[str] | None = None) -> tuple[str, int]:
    """Anthropic messages -> (transcript, images_omitted).

    Single text transcript (opencode takes plain text).
    Client tool blocks are rendered as compact text (no execution):
    tool_result -> ToolResult(id): <text, truncated 2000 chars>;
    tool_use -> ToolCall(name): <input JSON, truncated 500 chars>.
    image_status (from extract_request_images, aligned to _iter_image_blocks
    order): per-image-block "attached"|"omitted"|"invalid". Attached images
    render as an inline-file marker (the bytes travel as serve file parts);
    omitted/invalid render as notes and are counted. thinking blocks skipped.
    image_status=None preserves the legacy all-omitted behavior.
    """
    lines: list[str] = []
    images_omitted = 0
    img_idx = 0  # position into image_status (document order)

    def _image_marker() -> str:
        """Consume one image_status entry -> (marker text, counts as omitted)."""
        nonlocal img_idx, images_omitted
        st = (image_status[img_idx]
              if image_status is not None and img_idx < len(image_status)
              else "omitted")
        n = img_idx + 1
        img_idx += 1
        if st == "attached":
            return f"[image #{n} attached inline as file]"
        images_omitted += 1
        if st == "invalid":
            return f"[image #{n} skipped: invalid data]"
        return f"[image #{n} omitted: over slam limit]"

    for m in messages or []:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "")
        if isinstance(content, str):
            txt = content
        elif isinstance(content, list):
            chunks = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text" and b.get("text"):
                    chunks.append(b["text"])
                elif btype in ("image", "image_url"):
                    if image_status is None:
                        images_omitted += 1
                        chunks.append("[image omitted: no vision model available]")
                    else:
                        chunks.append(_image_marker())
                elif btype == "thinking":
                    continue  # never forward stale thinking blocks
                elif btype == "tool_result":
                    # Render images nested inside tool_result content inline.
                    c = b.get("content")
                    nested_marks = []
                    if isinstance(c, list):
                        for nb in c:
                            if isinstance(nb, dict) and nb.get("type") in ("image", "image_url"):
                                if image_status is None:
                                    images_omitted += 1
                                    nested_marks.append("[image omitted: no vision model available]")
                                else:
                                    nested_marks.append(_image_marker())
                    tid = b.get("tool_use_id") or "tool"
                    txt_result = _tool_result_text(b.get("content"))[:TOOL_RESULT_CHAR_LIMIT]
                    chunks.append(f"ToolResult({tid}): {txt_result}")
                    chunks.extend(nested_marks)
                elif btype == "tool_use":
                    name = b.get("name") or "tool"
                    try:
                        inp = json.dumps(b.get("input", {}), ensure_ascii=False)[:TOOL_INPUT_CHAR_LIMIT]
                    except Exception:
                        inp = str(b.get("input", ""))[:TOOL_INPUT_CHAR_LIMIT]
                    chunks.append(f"ToolCall({name}): {inp}")
            txt = "\n".join(chunks)
        else:
            txt = str(content)
        lines.append(f"{role}: {txt}")
    return "\n\n".join(lines).strip(), images_omitted


# ---------------- tool bridge (v2.0-beta: client executes, gateway decides) ----------------
TOOL_DECISION_ROLE = (
    "You are the reasoning engine behind an Anthropic-compatible assistant. "
    "You NEVER execute actions; you only decide the next response. "
    "Output STRICT JSON only (no markdown fences, no prose): "
    '{"content": [{"type": "text", "text": "..."}, '
    '{"type": "tool_use", "name": "<tool>", "input": {...}}]}. '
    "You may output text-only, tool_use-only, or mixed (text first, then tool calls). "
    "At most 4 tool_use blocks."
)

MAX_TOOL_USE_BLOCKS = 4


def _interpret_tool_choice(tool_choice) -> tuple[str, str | None]:
    """Return (mode, forced_name); mode in {auto, required, none, forced}."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        low = tool_choice.lower()
        if low == "none":
            return "none", None
        if low in ("any", "required"):
            return "required", None
        return "auto", None
    if isinstance(tool_choice, dict):
        t = str(tool_choice.get("type", "auto")).lower()
        if t == "none":
            return "none", None
        if t in ("any", "required"):
            return "required", None
        if t == "tool" and isinstance(tool_choice.get("name"), str) and tool_choice["name"]:
            return "forced", tool_choice["name"]
        return "auto", None
    return "auto", None


def _catalog_line(tool: dict) -> str:
    name = str(tool.get("name", "tool"))
    desc = str(tool.get("description", "") or "")[:300]
    schema = tool.get("input_schema")
    if isinstance(schema, dict):
        try:
            schema_s = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            schema_s = str(schema)
        schema_s = schema_s[:1500]
        return f"- {name}: {desc}. Input schema: {schema_s}"
    return f"- {name}: {desc}. Input schema: any object"


READ_ONLY_HINTS = ("read", "glob", "grep", "list", "find", "search", "fetch",
                   "ls", "tree", "view", "inspect", "open")
# Bookkeeping tools: they neither change the project nor count as exploration.
# They must NOT reset the read-only streak - a single TaskUpdate in the middle
# of a long explore used to reset the counter and hide the stall completely.
NEUTRAL_HINTS = ("taskupdate", "taskcreate", "todoread", "todowrite", "tasklist",
                 "listtasks", "exitplanmode", "exit_plan", "exitplan",
                 "askuserquestion", "websearch", "webfetch", "reportfindings")
# Real mutations: any of these means the agent is making progress.
MUTATING_HINTS = ("edit", "write", "notebookedit", "bash", "str_replace", "strreplace",
                  "apply_patch", "applypatch", "multiedit", "create", "delete",
                  "remove", "move", "rename", "mkdir", "save", "patch")
LOOP_REPEAT_THRESHOLD = 3  # same exact tool call this many times => loop
READ_ONLY_STREAK_THRESHOLD = 8  # this many read-only calls in a row => stall
# Distinct-file exploration stall: many reads, no mutation anywhere in the
# window. Catches the "read 14 different files and never act" spiral, which
# the exact-repeat and streak checks both miss.
EXPLORE_STALL_THRESHOLD = 6
LOOP_WINDOW = 12


def _canonical_tool_input(inp) -> str:
    try:
        return json.dumps(inp, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))[:500]
    except Exception:
        return str(inp)[:500]


def detect_tool_loop(messages: list) -> tuple[list[tuple[str, int]], int]:
    """Scan assistant tool_use blocks in the request history.

    Returns (repeated, read_only_streak): repeated is [(desc, count)]
    for exact tool calls (name + canonical input) seen >= threshold;
    read_only_streak is the trailing run of read-only calls.
    Stateless: everything comes from the history in this request.

    Only the last LOOP_WINDOW calls are examined: a repeat that happened
    20 turns ago is normal agent behaviour, not a loop.
    """
def _tool_kind(name: str) -> str:
    """'read' | 'mutate' | 'neutral' for a tool name (substring match, lowercase)."""
    low = name.lower()
    if any(h in low for h in MUTATING_HINTS):
        return "mutate"
    if any(h in low for h in READ_ONLY_HINTS):
        return "read"
    if any(h in low for h in NEUTRAL_HINTS):
        return "neutral"
    return "other"


def detect_tool_loop(messages: list) -> tuple[list[tuple[str, int]], int, int]:
    """Scan assistant tool_use blocks in the request history.

    Returns (repeated, read_only_streak, explore_stall):
      repeated       - [(desc, count)] for exact repeated calls (name+input)
      read_only_streak - trailing run of read-only calls; neutral bookkeeping
                         tools (TaskUpdate/TodoWrite/...) do NOT break it
      explore_stall  - how many read-only calls happened in the recent window
                         with no mutating call at all

    Only the last LOOP_WINDOW calls are examined: a repeat from 20 turns ago is
    normal agent behaviour, not a loop.
    """
    uses: list[tuple[str, str]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if (isinstance(b, dict) and b.get("type") == "tool_use"
                    and isinstance(b.get("name"), str)):
                inp = b.get("input", {})
                uses.append((b["name"],
                             _canonical_tool_input(inp if isinstance(inp, dict) else {})))
    window = uses[-LOOP_WINDOW:] if len(uses) > LOOP_WINDOW else uses
    counts: dict[tuple[str, str], int] = {}
    for key in window:
        counts[key] = counts.get(key, 0) + 1
    # Stable order: most frequent first, so the prompt shows the worst offender.
    repeated = sorted(([f"{name} {inp}", c] for (name, inp), c in counts.items()
                       if c >= LOOP_REPEAT_THRESHOLD), key=lambda x: -x[1])
    # Trailing read-only run. A neutral call (task bookkeeping) is skipped, not
    # treated as a break - otherwise one TaskUpdate hides a 20-call explore.
    streak = 0
    for name, _ in reversed(window):
        kind = _tool_kind(name)
        if kind == "read":
            streak += 1
        elif kind == "neutral":
            continue
        else:
            break
    # Window-wide stall: lots of reading, zero writing.
    kinds = [_tool_kind(n) for n, _ in window]
    read_total = sum(1 for k in kinds if k == "read")
    mutate_total = sum(1 for k in kinds if k == "mutate")
    explore_stall = read_total if mutate_total == 0 else 0
    return repeated, streak, explore_stall


_PATH_RE = re.compile(r"(?:^|[\s\"'(<])((?:[A-Za-z]:[\\/]|\\\\)[^\s\"'`<>|*?]{2,200})")
# Claude Desktop puts the attached project folder in the SYSTEM prompt as
# "Working directory: <path>" (inside an <env> block). It is NOT in the
# messages, which is why the model kept asking for a folder the user had
# already attached. We parse it out and treat it as the authoritative project.
_CWD_RE = re.compile(
    r"(?:working\s+directory|current\s+directory|project\s+(?:root|folder|directory)|"
    r"workspace\s+(?:root|folder|path)|cwd|folder|dossier|repertoire)"
    r"\s*[:=]\s*[\"']?((?:[A-Za-z]:[\\/]|\\\\)[^\"'\r\n<>]{1,200})",
    re.IGNORECASE)
# Same idea, but for a path that was pasted as a document/attachment header.
_ATTACH_RE = re.compile(
    r"(?:attached|attachment|folder|project|paste[ds]?|file)\s*[:=]\s*"
    r"[\"']?((?:[A-Za-z]:[\\/]|\\\\)[^\"'\r\n<>]{1,200})",
    re.IGNORECASE)


def discover_cwd_paths(system: str | list | None) -> list[str]:
    """Project folders the CLIENT already told us about, in priority order.

    This is the fix for "I attached the folder, why is it asking me again?":
    Claude Desktop announces the working directory in the system prompt, and we
    were dropping it on the floor. The result is authoritative - it beats any
    path the model might guess.
    """
    text = extract_system(system) if not isinstance(system, str) else system
    if not text:
        return []
    out: list[str] = []
    for rx in (_CWD_RE, _ATTACH_RE):
        for m in rx.finditer(text):
            p = m.group(1).strip().rstrip(".,;:)]}'\"")
            if SESSION_DIR.name.lower() in p.lower():
                continue  # our own sandbox is never the user's project
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                continue
            if p not in out:
                out.append(p)
        if out:
            break  # the explicit "working directory" wins over looser matches
    return out


def discover_project_paths(messages: list) -> list[str]:
    """Extract real Windows/UNC project paths the user actually mentioned.

    The model kept asking for a path it had ALREADY been given, then wandered
    into an unrelated sibling folder. It cannot tell a project path from a
    stray token inside a tool_result, so we do the extraction here and state
    the facts explicitly.
    """
    found: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    texts.append(b["text"])
                elif b.get("type") == "tool_use":
                    name = str(b.get("name", ""))
                    if name.lower() in ("read", "write", "edit", "notebookedit", "glob", "grep"):
                        v = b.get("input", {})
                        if isinstance(v, dict):
                            for key in ("file_path", "path", "notebook_path", "pattern"):
                                if isinstance(v.get(key), str):
                                    texts.append(v[key])
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, str):
                        texts.append(c[:400])
        for t in texts:
            for match in _PATH_RE.finditer(t):
                p = match.group(1).rstrip(".,;:)]}'\"")
                # Skip our own sandbox and image files: they are not the project.
                if SESSION_DIR.name.lower() in p.lower():
                    continue
                if p.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    continue
                if p not in found:
                    found.append(p)
    return found


def summarize_progress(messages: list) -> tuple[list[str], list[str], int]:
    """Extract what the agent ALREADY did, so it stops re-reading the same files.

    The model kept re-reading files it had already opened because the flat
    transcript never says "you already read this one". Free small models have
    no other memory: this summary is their only record of past actions.

    Returns (read_entries, written_entries, n_turns) where read_entries are
    "path (first N lines)" strings for Read/Glob/Grep calls.
    """
    reads: list[str] = []
    writes: list[str] = []
    seen_read: set[str] = set()
    seen_write: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            name = str(b.get("name", ""))
            inp = b.get("input", {})
            if not isinstance(inp, dict):
                continue
            path = (inp.get("file_path") or inp.get("notebook_path")
                    or inp.get("path") or "")
            if _tool_kind(name) == "read" and path and path not in seen_read:
                seen_read.add(path)
                reads.append(str(path))
            elif _tool_kind(name) == "mutate" and path and path not in seen_write:
                seen_write.add(path)
                writes.append(str(path))
    return reads, writes, len(reads)


def _progress_note(reads: list[str], writes: list[str]) -> str:
    """Render the already-done ledger the model can act on."""
    if not reads and not writes:
        return ""
    lines = ["Progress so far (from THIS conversation - do not repeat any of it):"]
    if reads:
        lines.append(f"  You already READ these {len(reads)} path(s) - do NOT read them again:")
        for p in reads[:25]:
            lines.append(f"    - {p}")
        if len(reads) > 25:
            lines.append(f"    ... and {len(reads) - 25} more")
    if writes:
        lines.append(f"  You already MODIFIED these {len(writes)} path(s) - they are done:")
        for p in writes[:15]:
            lines.append(f"    - {p}")
    return "\n".join(lines)


def _workspace_note(paths: list[str] | None = None) -> str:
    """Tell the model exactly where the project is and what it contains.

    Claude Desktop sends the whole conversation, but without grounding the
    model wanders into whatever folder it guesses, reports "it's empty"
    about an unrelated path, and invents a replacement file.
    """
    lines = [
        "Workspace facts (these are TRUE - do not re-discover them, do not guess):",
    ]
    wd = resolve_session_dir(paths)
    if paths and wd != SESSION_DIR:
        lines.append(
            f"- You are CURRENTLY WORKING INSIDE the user's project folder: {wd}. "
            f"That is your working directory for this whole task - relative paths "
            f"like \"main.py\" or \"src/app.py\" resolve there automatically.")
        lines.append(
            "- You ALREADY have read AND write access to that folder (same as a "
            "normal coding agent). Read, Edit, Write and run commands there "
            "directly. Do NOT claim you cannot access it and do NOT ask the user "
            "for the path - you are in it.")
    else:
        lines.append(
            f"- Your working directory is {SESSION_DIR} (a scratch sandbox), NOT "
            f"the user's project. {SESSION_DIR.name}/img/ holds only screenshots.")
    paths = paths or []
    if paths:
        lines.append(
            "- THE project folder for this task - the user already opened it in "
            "Claude Desktop. Use this exact path:")
        for p in paths[:5]:
            lines.append(f"    * {p}")
        lines.append(
            "- That folder is the ONLY scope. If any other path shows up in this "
            "conversation (a file read earlier, a folder merely mentioned, a sibling "
            "project), it is NOT the task: do not open it, do not treat it as the "
            "project, do not switch to it. Two paths here means one is stale "
            "context - keep using the folder listed above.")
        lines.append(
            "- To see what is inside, just Read/Glob there - you DO have access. "
            "Never ask the user for the path and never say you cannot find it.")
        lines.append(
            "- Read the project's real entry point first (the file the user named, "
            "or the main module that imports the rest). Style/design changes go "
            "in the EXISTING files - do NOT create a new .html or a parallel copy "
            "just because it is easier.")
    else:
        lines.append(
            "- The user has NOT given you a project path yet. If you need one, ASK "
            "for it in one short sentence. Never invent a path.")
    return "\n".join(lines)


def build_tool_decision_prompt(tools: list, tool_choice, transcript: str,
                               system_text: str, images_omitted: int,
                               max_tokens, images_attached: bool = False,
                               loop_repeats: list | None = None,
                               read_only_streak: int = 0,
                               explore_stall: int = 0,
                               project_paths: list | None = None,
                               done_reads: list | None = None,
                               done_writes: list | None = None) -> str:
    if isinstance(max_tokens, bool):
        budget = 2000
    elif isinstance(max_tokens, int):
        budget = min(max_tokens, 4000)
    else:
        budget = 2000
    mode, forced_name = _interpret_tool_choice(tool_choice)
    if mode == "auto":
        choice_line = "You may answer with text, call tools, or both."
    elif mode == "required":
        choice_line = "You MUST call at least one tool in this turn."
    elif mode == "forced":
        choice_line = f"You MUST call tool {forced_name} in this turn (first tool call must be {forced_name})."
    else:  # none (caller should have taken the text path; guard anyway)
        choice_line = "Answer with text only. Do not call tools."
    catalog = "\n".join(_catalog_line(t) for t in tools if isinstance(t, dict))
    parts = [TOOL_DECISION_ROLE, "", "Available tools:", catalog, "", choice_line, "",
             _workspace_note(project_paths), ""]
    # The already-done ledger goes BEFORE the conversation: the flat transcript
    # makes old reads look like fresh work, and this is the only thing that
    # tells the model which files it has already opened.
    ledger = _progress_note(done_reads or [], done_writes or []) if PROGRESS_LEDGER else ""
    if ledger:
        parts += [ledger, ""]
    parts.append("Conversation:")
    if images_attached:
        parts.append("Screenshots are attached inline; use them for coordinates/grounding.")
        if images_omitted > 0:
            parts.append(
                f"{images_omitted} image(s) omitted: over slam limit; "
                f"act conservatively for coordinate/grounding tasks needing those.")
    elif images_omitted > 0:
        parts.append(
            f"Images attached to tool results are NOT visible to you (vision unavailable); "
            f"{images_omitted} image block(s) were omitted. "
            f"Act conservatively for coordinate/grounding tasks.")
    parts.append(transcript if transcript else "(no conversation content)")
    if loop_repeats:
        shown = "; ".join(f"{desc} (x{c})" for desc, c in loop_repeats[:3])
        parts += ["",
                  f"LOOP WARNING: you already made these exact calls repeatedly: {shown}. "
                  f"Do NOT call them again. Take a DIFFERENT concrete action now "
                  f"(Edit/Write/Bash to change code, or write the final summary). "
                  f"Repeating an already-made call is forbidden."]
    if read_only_streak >= READ_ONLY_STREAK_THRESHOLD:
        parts += ["",
                  f"PROGRESS WARNING: your last {read_only_streak} tool calls were "
                  f"all read-only exploration. Stop exploring. Make the pending code "
                  f"change now, or summarize what is blocking you and what you need."]
    if explore_stall >= EXPLORE_STALL_THRESHOLD:
        parts += ["",
                  f"STOP EXPLORING: {explore_stall} of your last {min(len(tools) + explore_stall, LOOP_WINDOW)} "
                  f"tool calls only READ files - not one of them changed anything. "
                  f"You already know enough. Do NOT read, glob, grep or list anything else. "
                  f"This turn you MUST either (a) make the actual edit with Edit/Write, "
                  f"or (b) reply with a final text summary of what you found. "
                  f"Answering 'I will continue' and reading more files is forbidden."]
    if system_text:
        parts += ["", "System:", system_text]
    parts += ["", ("Keep text parts concise. Ensure every tool_use input is a JSON object "
                   f"matching the schema (all required fields present). max_tokens budget: {budget}.")]
    return "\n".join(parts)


def _strip_markdown_fences(s: str) -> str:
    t = s.strip()
    if not t.startswith("```"):
        return s
    lines = t.splitlines()
    lines = lines[1:]  # drop opening fence (``` or ```json)
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


_DRIVE_TOKEN_RE = re.compile(r"[A-Za-z]:(?:\\[^\s\"'`{}\\[\](),;]+)+")
# Double ONLY bare single backslashes (path separators like \A or \f).
# Never touch backslashes already escaping a quote/backslash (\" \\ \')
# — doubling those corrupts the JSON further.
_SINGLE_BACKSLASH_RE = re.compile(r"(?<!\\)\\(?![\"'\\])")
_BAD_ESCAPE_RE = re.compile(r"(?<!\\)\\(?![\"\\/bfnrtu])")


def _escape_drive_paths(s: str) -> str:
    """Double single backslashes inside Windows drive-path tokens.

    Free models often emit tool inputs like "H:\\AI\\faktino\\apps" with
    single backslashes (invalid JSON, and "\\f" would even parse as a
    formfeed). Doubling them inside drive-letter tokens preserves paths.
    Already-escaped ("\\\\") sequences are left alone.
    """

    def _dbl(m: "re.Match[str]") -> str:
        return _SINGLE_BACKSLASH_RE.sub(r"\\\\", m.group(0))

    return _DRIVE_TOKEN_RE.sub(_dbl, s)


def _lenient_json_loads(s: str) -> dict | None:
    """Strict JSON first; on failure repair model-typical damage and retry."""
    try:
        data = json.loads(s)
    except Exception:
        data = None
    if isinstance(data, dict):
        return data
    try:
        repaired = _BAD_ESCAPE_RE.sub(r"\\\\", _escape_drive_paths(s))
        data = json.loads(repaired, strict=False)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_decision_json(raw: str) -> dict | None:
    """Strict-ish extraction: raw -> fenced-stripped -> largest {...} span."""
    candidates: list[str] = []
    if isinstance(raw, str):
        candidates.append(raw.strip())
        fenced = _strip_markdown_fences(raw)
        if fenced != raw.strip():
            candidates.append(fenced)
    for cand in candidates:
        data = _lenient_json_loads(cand)
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            return data
    # Largest {...} span: balanced-brace scan, longest parseable first.
    if not isinstance(raw, str):
        return None
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(raw[start:i + 1])
                    start = -1
    # Also try first-{ to last-} as a candidate (covers nested cases).
    try:
        first, last = raw.index("{"), raw.rindex("}")
        if last > first:
            spans.append(raw[first:last + 1])
    except ValueError:
        pass
    spans.sort(key=len, reverse=True)
    for span in spans:
        data = _lenient_json_loads(span)
        if data is None:
            # Fences may wrap the span; strip and retry once.
            data = _lenient_json_loads(_strip_markdown_fences(span))
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            return data
    return None


def _validate_decision_items(data: dict, valid_names: set[str]) -> list[dict] | None:
    """Keep text/tool_use items; drop unknown tools; cap 4 tool_use. None = unusable."""
    content = data.get("content")
    if not isinstance(content, list):
        return None
    kept: list[dict] = []
    tool_count = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text" and isinstance(item.get("text"), str):
            kept.append({"type": "text", "text": item["text"]})
        elif t == "tool_use" and isinstance(item.get("name"), str):
            name = item["name"]
            if name not in valid_names:
                log.info("dropping unknown tool_use '%s' (not in request tool list)", name[:80])
                continue
            if tool_count >= MAX_TOOL_USE_BLOCKS:
                continue
            tool_count += 1
            inp = item.get("input", {})
            if not isinstance(inp, dict):
                inp = {}
            kept.append({"type": "tool_use", "name": name, "input": inp})
    if not kept:
        return None
    return kept


def _salvage_truncated_decision(raw: str, valid_names: set[str]) -> list[dict] | None:
    """Last resort for outputs cut off mid-JSON (e.g. max_tokens hit).

    Tries closing the truncated tail with a few suffixes and keeps the
    first result that parses AND contains at least one known tool_use.
    Returns None when nothing salvageable is found (caller falls back
    to plain text).
    """
    base = raw.rstrip()
    # A dangling trailing backslash would escape our closing quote.
    while len(base) >= 2 and base.endswith("\\") and not base.endswith("\\\\"):
        base = base[:-1]
    # Also try dropping an incomplete trailing fragment: rewind to the
    # last few complete-item boundaries so a cut mid-key/mid-string in
    # the LAST tool call still salvages the earlier complete calls.
    starts = [base]
    idx = len(base)
    for _ in range(3):
        idx = base.rfind("},", 0, idx - 1)
        if idx < 0:
            break
        starts.append(base[:idx + 1])
    suffixes = ('', '"', '}', ']}', '"]}', '"}', '"}]}', '"}}]}',
                '}]}', '}}]}')
    for start in starts:
        for suffix in suffixes:
            data = _lenient_json_loads(start + suffix)
            if not isinstance(data, dict):
                continue
            items = _validate_decision_items(data, valid_names)
            if items and any(i["type"] == "tool_use" for i in items):
                return items
    return None


def _assign_tool_ids(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        if item["type"] == "tool_use":
            out.append({"type": "tool_use",
                        "id": "toolu_" + uuid.uuid4().hex[:12],
                        "name": item["name"], "input": item["input"]})
        else:
            out.append(item)
    return out


def maybe_capture_request(body: dict, tools: list) -> None:
    """Sanitized capture for debugging (GATEWAY_CAPTURE=1). No secrets, truncated."""
    if not GATEWAY_CAPTURE:
        return
    try:
        cap_dir = BASE_DIR / "captures"
        cap_dir.mkdir(exist_ok=True, parents=True)
        tool_info = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            schema = t.get("input_schema")
            if isinstance(schema, dict):
                try:
                    schema_s = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))[:1500]
                except Exception:
                    schema_s = str(schema)[:1500]
            else:
                schema_s = "any object"
            tool_info.append({"name": str(t.get("name", "")),
                              "description": str(t.get("description", "") or "")[:300],
                              "input_schema": schema_s})

        def _san_block(b):
            if not isinstance(b, dict):
                return {"type": "other"}
            bt = b.get("type")
            if bt == "text":
                return {"type": "text", "text": str(b.get("text", ""))[:300]}
            if bt == "image":
                try:
                    src = b.get("source", {})
                    data_s = src.get("data", "") if isinstance(src, dict) else ""
                    n = len(data_s) if isinstance(data_s, str) else "?"
                except Exception:
                    n = "?"
                return {"type": "image", "note": f"[image {n} bytes]"}
            if bt == "tool_use":
                try:
                    inp_s = json.dumps(b.get("input", {}), ensure_ascii=False)[:300]
                except Exception:
                    inp_s = str(b.get("input", ""))[:300]
                return {"type": "tool_use", "name": str(b.get("name", "")),
                        "id": str(b.get("id", ""))[:64], "input": inp_s}
            if bt == "tool_result":
                c = b.get("content")
                if isinstance(c, str):
                    sc = c[:300]
                elif isinstance(c, list):
                    chunks = []
                    for nb in c:
                        if isinstance(nb, dict) and nb.get("type") == "image":
                            chunks.append("[image ? bytes]")
                        elif isinstance(nb, dict) and isinstance(nb.get("text"), str):
                            chunks.append(nb["text"][:300])
                        elif isinstance(nb, str):
                            chunks.append(nb[:300])
                    sc = "\n".join(chunks)[:600]
                else:
                    sc = str(c)[:300] if c is not None else ""
                return {"type": "tool_result",
                        "tool_use_id": str(b.get("tool_use_id", ""))[:64], "content": sc}
            return {"type": str(bt)[:32]}

        msgs = []
        for m in (body.get("messages") or []) if isinstance(body, dict) else []:
            if not isinstance(m, dict):
                continue
            c = m.get("content", "")
            if isinstance(c, str):
                sc = c[:300]
            elif isinstance(c, list):
                sc = [_san_block(b) for b in c]
            else:
                sc = str(c)[:300]
            msgs.append({"role": str(m.get("role", ""))[:16], "content": sc})
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "model": str(body.get("model", ""))[:80],
                   "tool_choice": str(body.get("tool_choice", ""))[:160],
                   "max_tokens": body.get("max_tokens"),
                   "tools": tool_info, "messages": msgs}
        fname = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6] + ".json"
        (cap_dir / fname).write_text(json.dumps(payload, ensure_ascii=False)[:20000],
                                     encoding="utf-8")
        log.info("captured sanitized request -> captures/%s", fname)
    except Exception as e:
        log.info("capture failed (%s)", type(e).__name__)


def check_gateway_auth(request: Request) -> str | None:
    """Return None if authorized, else an error string (never echoes the key)."""
    auth = request.headers.get("authorization", "")
    xkey = request.headers.get("x-api-key", "")
    got = ""
    if xkey:
        got = xkey
    elif auth.lower().startswith("bearer "):
        got = auth[7:].strip()
    if not got or not secrets.compare_digest(got, GATEWAY_API_KEY):
        return "invalid API key"
    return None


def anthropic_error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": "api_error", "message": message}},
                        status_code=status)


def sse_event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def anthropic_sse_blocks(blocks: list[dict], alias: str, msg_id: str,
                         usage: dict, stop_reason: str) -> StreamingResponse:
    """SSE for tool-bridged replies: one content_block per text/tool_use block."""
    async def gen():
        yield sse_event("message_start", {"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": alias,
            "content": [], "stop_reason": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}}})
        for i, b in enumerate(blocks):
            if b.get("type") == "tool_use":
                yield sse_event("content_block_start", {
                    "type": "content_block_start", "index": i,
                    "content_block": {"type": "tool_use", "id": b["id"],
                                      "name": b["name"], "input": {}}})
                try:
                    s = json.dumps(b.get("input", {}), ensure_ascii=False)
                except Exception:
                    s = "{}"
                # Break input JSON into 2-3 input_json_delta chunks.
                n = 3 if len(s) > 200 else 2
                k = max(1, (len(s) + n - 1) // n)
                for j in range(0, len(s), k):
                    yield sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": i,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": s[j:j + k]}})
                yield sse_event("content_block_stop",
                                {"type": "content_block_stop", "index": i})
            else:
                yield sse_event("content_block_start", {
                    "type": "content_block_start", "index": i,
                    "content_block": {"type": "text", "text": ""}})
                yield sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": i,
                    "delta": {"type": "text_delta", "text": b.get("text", "")}})
                yield sse_event("content_block_stop",
                                {"type": "content_block_stop", "index": i})
        yield sse_event("message_delta", {"type": "message_delta",
                                          "delta": {"stop_reason": stop_reason},
                                          "usage": {"output_tokens": usage.get("output_tokens", 0)}})
        yield sse_event("message_stop", {"type": "message_stop"})
    return StreamingResponse(gen(), media_type="text/event-stream")


def anthropic_sse(text: str, alias: str, msg_id: str, usage: dict) -> StreamingResponse:
    async def gen():
        yield sse_event("message_start", {"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": alias,
            "content": [], "stop_reason": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}}})
        yield sse_event("content_block_start", {"type": "content_block_start", "index": 0,
                                                "content_block": {"type": "text", "text": ""}})
        # Upstream is non-streaming: re-emit full text as one delta (documented).
        yield sse_event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                "delta": {"type": "text_delta", "text": text}})
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield sse_event("message_delta", {"type": "message_delta",
                                          "delta": {"stop_reason": "end_turn"},
                                          "usage": {"output_tokens": usage.get("output_tokens", 0)}})
        yield sse_event("message_stop", {"type": "message_stop"})
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------- blocking workers (run in a threadpool, never on the loop) --
# Every upstream call is synchronous (httpx.Client + blocking reads). Running
# it directly inside an `async def` endpoint blocks the whole event loop, so
# /health stops answering and concurrent requests serialise behind one slow
# model. These helpers are pure-sync and are dispatched with run_in_threadpool.
def _chat_worker(alias: str, system: str, prompt: str,
                 img_files: list[dict] | None, deadline: float,
                 work_dir: Path | None = None) -> tuple[str, str, dict]:
    with _new_client(timeout=UPSTREAM_TIMEOUT) as client:
        return chat_with_fallback(client, alias, system, prompt,
                                  image_paths=img_files or None, deadline=deadline,
                                  work_dir=work_dir)


def _decision_worker(alias: str, decision_prompt: str, retry_prompt: str | None,
                     img_files: list[dict] | None, valid_names: set[str],
                     forced_name: str | None, deadline: float,
                     work_dir: Path | None = None
                     ) -> tuple[str, str, dict, list[dict]]:
    """Run the tool-bridge decision (and optional forced retry) off-loop.

    Returns (raw, used_model, usage, items). items is already salvaged or
    text-wrapped, so the caller only has to assign ids and respond.
    """
    with _new_client(timeout=UPSTREAM_TIMEOUT) as client:
        raw, used_model, usage = chat_with_fallback(
            client, alias, "", decision_prompt,
            image_paths=img_files or None, deadline=deadline, work_dir=work_dir)
        data = _extract_decision_json(raw)
        items = _validate_decision_items(data, valid_names) if data is not None else None
        tool_names = [i["name"] for i in items if i["type"] == "tool_use"] if items else []
        # Forced-tool second attempt with a stronger forcing line. Skipped
        # when the budget is already spent (it would just burn the deadline).
        if retry_prompt and forced_name and forced_name not in tool_names:
            if deadline is not None and time.time() >= deadline:
                log.info("forced retry skipped: time budget exhausted")
                retry_prompt = None
        if retry_prompt and forced_name:
            log.info("forced tool '%s' absent in first decision; retrying with stronger force",
                     forced_name[:80])
            try:
                raw2, used_model2, usage2 = chat_with_fallback(
                    client, alias, "", retry_prompt,
                    image_paths=img_files or None, deadline=deadline,
                    work_dir=work_dir)
                data2 = _extract_decision_json(raw2)
                items2 = (_validate_decision_items(data2, valid_names)
                          if data2 is not None else None)
                tool_names2 = [i["name"] for i in items2
                               if i["type"] == "tool_use"] if items2 else []
                if items2 is not None and forced_name in tool_names2:
                    raw, used_model, usage = raw2, used_model2, usage2
                    items, tool_names = items2, tool_names2
                else:
                    log.info("forced tool '%s' still absent; text fallback (never 502)",
                             forced_name[:80])
                    items = [{"type": "text", "text": raw2[:4000]}]
            except UpstreamError as e2:
                # Retry failed but first attempt may still be usable.
                log.info("forced retry upstream failed (%s); using first attempt",
                         str(e2)[:120])
        if items is None:
            salvaged = _salvage_truncated_decision(raw, valid_names)
            if salvaged is not None:
                log.info("salvaged truncated tool decision (%d tool_use) via %s",
                         sum(1 for i in salvaged if i["type"] == "tool_use"),
                         used_model)
                items = salvaged
            else:
                full = raw.encode("unicode_escape").decode("ascii", "replace")
                log.info("tool-decision parse failed (%d chars, model %s): %s",
                         len(raw), used_model, full[:8000])
                items = [{"type": "text", "text": raw[:4000]}]
    return raw, used_model, usage, items


# ---------------- app ----------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Modern replacement for on_event (deprecated in FastAPI 0.115+)."""
    _startup()
    try:
        yield
    finally:
        _shutdown()


app = FastAPI(title="opencode-free-tier gateway", lifespan=lifespan)


def _prepare_vision(messages: list, alias: str) -> tuple[str, list[dict], int, bool, list]:
    """Extract images + build transcript for one request.

    Returns (prompt, attach_files, omitted_total, images_attached, compacted).
    attach_files is non-empty ONLY when images were saved AND this alias
    has a vision-ranked model; otherwise text-only with omission notes
    (legacy behavior). omitted_total counts over-limit + invalid + dropped.

    Long agentic histories are compacted first (first message = goal,
    plus a summary note, plus the last N turns) so late turns don't
    balloon past upstream timeouts. Only images in the kept turns are
    attached (old screenshots are dropped, not re-sent every turn).

    The compacted list is returned so callers run other history analysis
    (loop detection) on exactly what the model will see.
    """
    msgs, hist_omitted, hist_tools = compact_history(messages)
    if hist_omitted:
        log.info("history: compacted %d earlier turn(s) (tools there: %s)",
                 hist_omitted, ",".join(hist_tools[:10]) or "none")
    files, over, invalid, status = extract_request_images(msgs)
    dropped = 0
    vision_first = [m for m in vision_rank() if m in COMBO.get(alias, [])]
    attach = bool(files) and bool(vision_first)
    if files and not attach:
        status = ["omitted" if s == "attached" else s for s in status]
        dropped = len(files)
        files = []
        log.info("vision: %d image(s) dropped (no vision model in alias '%s')",
                 dropped, alias)
    # flatten_messages already emits a per-image marker for every omitted
    # block, so no extra "[N images omitted]" line is appended here (it
    # would double-count the same omissions for the model).
    prompt, _flat_omitted = flatten_messages(msgs, image_status=status)
    prompt = _cap_text(prompt)
    # Best-effort disk hygiene: drop stale images from previous requests.
    try:
        cleanup_images()
    except Exception:
        pass
    return prompt, files, over + invalid + dropped, attach, msgs


HISTORY_KEEP_LAST = _int_env("GATEWAY_HISTORY_KEEP_LAST", 60)
TRANSCRIPT_CHAR_CAP = _int_env("GATEWAY_TRANSCRIPT_CHAR_CAP", 400000)


def _cap_text(s: str, limit: int = 0) -> str:
    """Hard cap on transcript length (keep head + tail)."""
    lim = limit or TRANSCRIPT_CHAR_CAP
    if len(s) <= lim:
        return s
    # head must stay strictly smaller than lim, otherwise s[-(lim - head):]
    # slices from the wrong end (a 0 or negative tail silently re-returns
    # the whole string, blowing past the cap).
    head = min(4000, max(0, lim // 2))
    tail = max(0, lim - head)
    if tail == 0:
        return s[:lim]
    return (s[:head] +
            f"\n\n[... {len(s) - lim} chars of middle history omitted ...]\n\n" +
            s[-tail:])


def compact_history(messages: list) -> tuple[list, int, list]:
    """Shrink long histories: [first (=goal), summary note, last N turns].

    Returns (compacted, omitted_count, tool_names_in_omitted).
    Short histories pass through untouched.
    """
    msgs = [m for m in messages or [] if isinstance(m, dict)]
    if len(msgs) <= HISTORY_KEEP_LAST + 1:
        return msgs, 0, []
    omitted = msgs[1:-HISTORY_KEEP_LAST]
    tools: list[str] = []
    for m in omitted:
        c = m.get("content", "")
        if not isinstance(c, list):
            continue
        for b in c:
            if (isinstance(b, dict) and b.get("type") == "tool_use"
                    and isinstance(b.get("name"), str)
                    and b["name"] not in tools):
                tools.append(b["name"])
    note = {"role": "user", "content": (
        f"[{len(omitted)} earlier turns omitted for brevity. "
        f"Tools used in omitted turns: {', '.join(tools[:10]) or 'none'}. "
        f"Continue from the recent context below.]")}
    return [msgs[0], note] + msgs[-HISTORY_KEEP_LAST:], len(omitted), tools


def _startup():
    SESSION_DIR.mkdir(exist_ok=True)  # sandbox for model tool calls
    IMG_DIR.mkdir(exist_ok=True)  # v3: inbound images for vision routing
    log.info("combo aliases: %s", {k: len(v) for k, v in COMBO.items()})
    log.info("vision rank (%s): %s (max %d images, %.1f MB total)",
             VISION_PROBE_DATE, vision_rank(),
             MAX_IMAGES, MAX_IMAGE_BYTES / 1024 / 1024)
    log.info("limits: per-model %.0fs, per-request %.0fs, fail-cooldown %ds, img-ttl %.1fh",
             UPSTREAM_TIMEOUT, TOTAL_BUDGET_S, FAIL_COOLDOWN_S, IMG_TTL_H)
    try:
        cleanup_images()
    except Exception:
        pass
    if not ensure_serve_running():
        log.error("opencode serve NOT available - /health will report it, chat will 502")


def _shutdown():
    # Stop the serve instance WE spawned (if any); leave pre-existing ones alone.
    global _serve_proc
    if _serve_proc is not None and _serve_proc.poll() is None:
        log.info("stopping spawned opencode serve (pid %s)", _serve_proc.pid)
        _serve_proc.terminate()
        try:
            _serve_proc.wait(timeout=10)
        except Exception:
            _serve_proc.kill()
        _serve_proc = None


@app.get("/health")
def health():
    ok, detail = serve_probe()
    return {
        "gateway": "ok",
        "uptime_s": int(time.time() - STARTED_AT),
        "serve_url": OPENCODE_SERVE_URL,  # host only; creds never exposed
        "serve_reachable": ok,
        "serve_detail": detail,
        "aliases": {k: len(v) for k, v in COMBO.items()},
        "tools_lockdown": DISABLE_TOOLS,
        "vision_rank": vision_rank(),
        "vision_probe_date": VISION_PROBE_DATE,
        "max_images": MAX_IMAGES,
        "max_image_mb": round(MAX_IMAGE_BYTES / 1024 / 1024, 2),
    }


@app.get("/v1/models")
def list_models(request: Request):
    err = check_gateway_auth(request)
    if err:
        return JSONResponse({"error": {"message": err, "type": "invalid_api_key"}}, status_code=401)
    # OpenAI format so Claude Desktop auto-discovers the 3 aliases.
    return {"object": "list", "data": [
        {"id": alias, "object": "model", "created": int(STARTED_AT), "owned_by": "combo"}
        for alias in COMBO
    ]}


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    err = check_gateway_auth(request)
    if err:
        return anthropic_error(401, err)
    try:
        body = await request.json()
    except Exception:
        return anthropic_error(400, "invalid JSON body")
    alias = body.get("model", "")
    if alias not in COMBO:
        return anthropic_error(404, f"unknown model '{alias}'; see GET /v1/models")
    tools = body.get("tools")
    has_tools = isinstance(tools, list) and len(tools) > 0
    tool_choice = body.get("tool_choice", None)
    mode, forced_name = _interpret_tool_choice(tool_choice)
    # Silently accept/ignore Cowork extras: betas, metadata,
    # service_tier, output_config (no-op).
    system = extract_system(body.get("system"))
    stream = body.get("stream") is True
    msg_id = "msg_" + uuid.uuid4().hex[:24]

    if not has_tools or mode == "none":
        # No-tools path: text, or vision-first when images are attached.
        if has_tools and mode == "none":
            log.info("tool_choice none -> text-only (%d tools ignored)", len(tools))
        prompt, img_files, _omitted, _attached, _msgs = _prepare_vision(
            body.get("messages", []), alias)
        if not prompt and not system:
            return anthropic_error(400, "empty prompt: provide messages or system")
        deadline = time.time() + TOTAL_BUDGET_S
        wd = resolve_session_dir(discover_cwd_paths(body.get("system")) or
                                 discover_project_paths(body.get("messages", [])))
        if PROJECT_LOG:
            log.info("project: work_dir=%s (no-tools path)", wd)
        try:
            text, used_model, usage = await run_in_threadpool(
                _chat_worker, alias, system, prompt or system, img_files, deadline, wd)
        except UpstreamError as e:
            return anthropic_error(502, f"upstream failed: {str(e)[:300]}")
        if stream:
            return anthropic_sse(text, alias, msg_id, usage)
        return {
            "id": msg_id, "type": "message", "role": "assistant", "model": alias,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": usage.get("input_tokens", 0),
                      "output_tokens": usage.get("output_tokens", 0)},
        }

    # ---- Tool-bridge path: never 400 for tools. ----
    valid_names = {str(t.get("name")) for t in tools
                   if isinstance(t, dict) and isinstance(t.get("name"), str) and t.get("name")}
    if mode == "forced" and forced_name not in valid_names:
        log.info("forced tool '%s' not in request tool list; treating as auto",
                 str(forced_name)[:80])
        mode, forced_name = "auto", None
    transcript, img_files, images_omitted, images_attached, _kept = _prepare_vision(
        body.get("messages", []), alias)
    maybe_capture_request(body, tools)
    # Loop detection runs on the FULL history, not the compacted one. The guard
    # is a safety signal about the agent's behaviour, so dropping early turns
    # would hide exactly the long explore-stall we need to catch (compaction
    # keeps only the last HISTORY_KEEP_LAST messages).
    loop_repeats, ro_streak, explore_stall = detect_tool_loop(body.get("messages", []))
    if loop_repeats or ro_streak >= READ_ONLY_STREAK_THRESHOLD \
            or explore_stall >= EXPLORE_STALL_THRESHOLD:
        log.info("loop-guard: %d repeated call(s), read-only streak %d, explore-stall %d",
                 len(loop_repeats), ro_streak, explore_stall)
    all_msgs = body.get("messages", [])
    # The folder the user ATTACHED in Claude Desktop comes from the system
    # prompt, not the chat. It is authoritative: when present, it is the ONLY
    # project path we state. Paths seen in the chat are usually stale context
    # (a folder mentioned once, a sibling project, an old file) - feeding them
    # to the model as peers is what made it wander into the wrong folder.
    project_paths = discover_cwd_paths(body.get("system"))
    if not project_paths:
        for p in discover_project_paths(all_msgs):
            if p not in project_paths:
                project_paths.append(p)
    done_reads, done_writes, _n_reads = summarize_progress(all_msgs)
    # Where the model will actually run. This is what makes "I entered your
    # folder" true: the serve session is created IN this directory, so the
    # model's own tools (Read/Glob/Edit) operate on the user's project.
    work_dir = resolve_session_dir(project_paths)
    if PROJECT_LOG:
        log.info("project: paths=%s work_dir=%s reads=%d",
                 project_paths or "[]", work_dir, len(done_reads))
    decision_prompt = build_tool_decision_prompt(
        [t for t in tools if isinstance(t, dict)], tool_choice,
        transcript, system, images_omitted, body.get("max_tokens"),
        images_attached=images_attached,
        loop_repeats=loop_repeats, read_only_streak=ro_streak,
        explore_stall=explore_stall,
        project_paths=project_paths,
        done_reads=done_reads, done_writes=done_writes)
    # Forced-tool retry is prepared here but executed inside the worker, so
    # the "one whole request" deadline covers BOTH attempts.
    retry_prompt = None
    if mode == "forced" and forced_name in valid_names:
        retry_prompt = (decision_prompt +
                        f"\nCRITICAL: your previous reply did not call {forced_name}; "
                        f"reply again with ONLY the JSON calling {forced_name}.")
    deadline = time.time() + TOTAL_BUDGET_S
    try:
        raw, used_model, usage, items = await run_in_threadpool(
            _decision_worker, alias, decision_prompt, retry_prompt,
            img_files, valid_names, forced_name if retry_prompt else None,
            deadline, work_dir)
    except UpstreamError as e:
        # 502 only if ALL combo models failed.
        return anthropic_error(502, f"upstream failed: {str(e)[:300]}")
    blocks = _assign_tool_ids(items)
    has_use = any(b["type"] == "tool_use" for b in blocks)
    stop_reason = "tool_use" if has_use else "end_turn"
    names_s = ",".join(b["name"] for b in blocks if b["type"] == "tool_use")
    if has_use:
        log.info("bridged %d tools -> %d tool_use (%s) via %s",
                 len(tools), sum(1 for b in blocks if b["type"] == "tool_use"),
                 names_s[:200], used_model)
    else:
        log.info("bridged %d tools -> text-only via %s", len(tools), used_model)
    if stream:
        return anthropic_sse_blocks(blocks, alias, msg_id, usage, stop_reason)
    return {
        "id": msg_id, "type": "message", "role": "assistant", "model": alias,
        "content": blocks,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": usage.get("input_tokens", 0),
                  "output_tokens": usage.get("output_tokens", 0)},
    }


@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    """OpenAI-compatible endpoint (same combos), stream + non-stream."""
    err = check_gateway_auth(request)
    if err:
        return JSONResponse({"error": {"message": err, "type": "invalid_api_key"}}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
    alias = body.get("model", "")
    if alias not in COMBO:
        return JSONResponse({"error": {"message": f"unknown model '{alias}'"}}, status_code=404)
    system, convo = "", []
    for m in body.get("messages", []) or []:
        if m.get("role") == "system":
            c = m.get("content", "")
            system += c if isinstance(c, str) else extract_system(c)
        else:
            convo.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    prompt, img_files, _omitted, _attached, _msgs = _prepare_vision(convo, alias)
    if not prompt and not system:
        return JSONResponse({"error": {"message": "empty prompt"}}, status_code=400)
    stream = body.get("stream") is True
    chat_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    deadline = time.time() + TOTAL_BUDGET_S
    wd = resolve_session_dir(discover_cwd_paths(system) or
                             discover_project_paths(body.get("messages", [])))
    if PROJECT_LOG:
        log.info("project: work_dir=%s (openai path)", wd)
    try:
        text, used_model, usage = await run_in_threadpool(
            _chat_worker, alias, system, prompt or system, img_files, deadline, wd)
    except UpstreamError as e:
        return JSONResponse({"error": {"message": f"upstream failed: {str(e)[:300]}"}}, status_code=502)
    if not stream:
        return {
            "id": chat_id, "object": "chat.completion", "created": created, "model": alias,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                      "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)},
        }

    async def gen():
        head = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                "model": alias, "choices": [{"index": 0, "delta": {"role": "assistant",
                                                                  "content": text},
                                             "finish_reason": None}]}
        yield "data: " + json.dumps(head, ensure_ascii=False) + "\n\n"
        tail = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                "model": alias, "choices": [{"index": 0, "delta": {},
                                             "finish_reason": "stop"}]}
        yield "data: " + json.dumps(tail) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
