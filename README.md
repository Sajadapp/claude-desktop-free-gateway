# Claude Desktop Gateway — راهنمای کوتاه / Short guide

## Start / اجرا

Double-click `run_gateway.bat` (or run it in a terminal).
It activates `.venv`, starts the gateway on `http://127.0.0.1:3457`,
and auto-spawns `opencode serve` on `:4097` if needed.

دابل‌کلیک روی `run_gateway.bat`. گیت‌وی روی `http://127.0.0.1:3457`
بالا می‌آید و اگر `opencode serve` روی `:4097` نباشد خودش اجرایش می‌کند.

## Claude Desktop settings / تنظیمات

- Gateway base URL: `http://127.0.0.1:3457`
- Auth scheme: `bearer`
- Gateway API key: paste the value of `GATEWAY_API_KEY` from `.env`
  (default `local-dev-key-12345` — change it to your own secret).
- Model list (auto-discovered from `GET /v1/models`):
  `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`
  (all three map to the SAME free-model combo in `combo.json`).

## Reorder combo / ترتیب مدل‌ها

Edit `combo.json`: each alias maps to an ordered list.
The gateway tries them in order and falls back on error / rate-limit /
empty reply. Put chat-compatible models first, responses-only ones last.
Restart the gateway after editing.

## Troubleshooting / رفع اشکال

- `401 invalid API key`: the key in Claude Desktop ≠ `GATEWAY_API_KEY`
  in `.env`. Copy it exactly.
- `502 upstream failed` / `429`: a combo model hit rate-limit or errored;
  the gateway already tried the next ones. Check `GET /health`
  (`serve_reachable` must be `true`) and try again later.
- Serve `401 auth-rejected`: `OPENCODE_SERVE_USERNAME/PASSWORD` in `.env`
  don't match the running serve instance. Stop the stray
  `opencode serve` (or fix `.env`) and restart the gateway.
- NOTE: requests are slowish (free models via serve: tens of seconds) and
  each request burns ~20k input tokens of opencode system prompt.

## Limits (v1) / محدودیت‌ها

Text chat only. `tools`/computer-use -> clear `400`. Images -> replaced
with a placeholder note. `thinking` blocks are not forwarded/returned.
Streaming endpoints exist but re-emit the full upstream reply as SSE
(upstream `serve` call itself is non-streaming).
Model tool calls (bash/read/...) stay ENABLED (disabling them breaks free
models) but sessions run isolated in `./sandbox/` — keep no secrets there.
Stop the gateway with Ctrl+C (graceful: also stops its spawned serve).

## FAQ: Cowork / computer-use tools (v2.0-beta)

Cowork/Code agentic loop now functions: when a request carries `tools`,
the gateway builds a TOOL-DECISION prompt, asks the upstream free-model
combo to decide, and returns valid Anthropic `text` / `tool_use` blocks.
Claude Desktop (the client) executes the tools; the gateway only decides
— it never runs actions itself. Multi-step loops work: each turn carries
full history (`tool_use` + `tool_result`), stateless on the gateway side.
File/bash-style tools that the client executes work; `tool_choice`
`auto` / `any` / `none` / forced-tool are honored (forced retries once,
then falls back to text, never 502 for a missed force).

What does NOT work well: computer-use coordinate/grounding tasks are
unreliable on text-only fallback models — but when screenshots are
attached the gateway now routes vision-capable models first (see
"Vision / screenshots" below). Latency is ~30-90s per turn
(free models via serve). Destructive actions are a real risk: the model
may hallucinate paths/arguments.

## Vision / screenshots (v3.0-beta, verified 2026-09-24)

Anthropic image blocks (base64 PNG/JPEG, top-level or nested in
`tool_result`) are decoded, magic-byte verified, capped
(`GATEWAY_MAX_IMAGES=5`, `GATEWAY_MAX_IMAGE_MB=4` total per request),
saved under `sandbox/img/`, and attached to the upstream serve call as
file parts (`{"type":"file","mime":...,"url":"file://..."}`). Extras are
dropped with a `[N image(s) omitted: over slam limit]` prompt note;
corrupt blocks are skipped gracefully.

Verified vision matrix (probe: 400x200 red PNG reading "ABC123",
prompt `Describe exactly what you see...`):

| model | vision | latency | note |
|---|---|---|---|
| opencode/mimo-v2.6-flash-free | PASS | ~27s | perfect description |
| opencode-go/muse-spark-1.3-contributor | PASS | ~9.5s | accurate; vision-rank #1 since 2026-09-25 |
| opencode/muse-spark-1.2-contributor-free | PASS | ~10s | accurate; rank #2 |
| opencode/muse-spark-1.3-contributor-free | PASS | ~11s | accurate; rank #3 |
| opencode/space-bunny-free | PASS | ~8s | correct gist, minor border/color quibbles; rank #4 |
| opencode/ling-3.0-flash-fin-free | FAIL | ~4s | explicit "does not support image input" denial |
| opencode/nemotron-3-ultra-free | FAIL | ~4s | explicit vision denial |
| opencode/nemotron-3.5-lightning-free | FAIL | ~32s | vision denial (slow) |
| opencode/mimo-v2.5-free | UNSUPPORTED | instant 500 | HTTP 500 on image AND text-only; model broken server-side |

Routing: with images present, PASS models are tried FIRST in rank
order, then remaining combo models as text-only fallback (omission
note appended). Without images the `combo.json` order is untouched.
The tool-decision prompt drops the blindness disclaimer when images
are attached (`Screenshots are attached inline; use them for
coordinates/grounding.`) and keeps it when images were omitted or no
vision model exists. Answering model + `vision: k images -> trying
<model> (vision-rank #n)` are in the gateway log.

Re-rank without code edit: set `VISION_RANK_ORDER` in `.env`
(comma-separated refs, e.g. `opencode/space-bunny-free,opencode/mimo-v2.6-flash-free`).
`combo.json` stays the text-order source of truth.

Residual limits (honest): small free models ground coordinates only
roughly (test: clicking "ABC123" in a 400x200 probe returned a
plausible but unverified [138,93]); screenshots are forwarded at
original resolution (no downscaling — large shots count against the
4MB cap); JPEG vs PNG both accepted; `mimo-v2.5-free` 500s on
everything (excluded from vision AND unreliable for text).

Safety guidance: supervise every tool call; start with read-only tasks;
keep work inside the Claude Desktop scope; do not grant destructive
tools until read-only loops are stable.

Debug: set `GATEWAY_CAPTURE=1` in `.env` (default `0`) and restart — the
gateway saves a sanitized request (model, tool names+schemas truncated,
message text truncated to 300 chars, images replaced with
`[image N bytes]`) per tool-bridged turn into `captures/` with a
timestamp name. Send that file when reporting a failing payload.
