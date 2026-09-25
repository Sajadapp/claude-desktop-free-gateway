# `opencode serve` API notes (probed 2026-09-24, opencode 1.18.32)

Auth: HTTP Basic, user = `$OPENCODE_SERVER_USERNAME` (default `opencode`),
password = `$OPENCODE_SERVER_PASSWORD`. Every call below needs it.
Base: `http://127.0.0.1:4097`. OpenAPI doc: `GET /doc` (auth required).

## Minimal chat flow (verified working with free tier)

1. Create session (stateless: one per chat request):
   `POST /session?directory=<dir>` body `{"title": "gateway chat"}`
   -> `200 {"id": "ses_...â", ...}`.
2. Send prompt (synchronous, returns full JSON - NOT SSE):
   `POST /session/{id}/message` body:
   ```json
   {
     "model": {"providerID": "opencode", "modelID": "mimo-v2.6-flash-free"},
     "system": "<optional system prompt>",
     "parts": [{"type": "text", "text": "<prompt>"}],
     "tools": {"<toolName>": false, "...": false}
   }
   ```
   -> `200 {"info": {"role": "assistant", "modelID": "...", "finish": "stop",
   "tokens": {"input": N, "output": M}}, "parts": [...]}`.
3. Extract reply: concatenate `parts[]` where `type == "text"` (`-> .text`).
   Other part types to ignore: `step-start`, `step-finish`, `reasoning`,
   `tool` (when tools enabled).

## Verified facts

- Free model `opencode/mimo-v2.6-flash-free` answered via serve in ~33s
  (`hello from free tier` test). Direct Zen REST for the same model gives
  `403 FreeTierError`, so serve is mandatory for free models.
- `POST /session/{id}/message` is blocking JSON despite the doc description
  ("streaming the AI response"); live SSE comes from `GET /event`
  (`text/event-stream`, global). Gateway v1 uses non-stream upstream and
  re-emits Anthropic/OpenAI SSE itself.
- `model` can be set per-message; no need to set it at session creation.
- `tools` map disables tools per prompt (`false` = off). Tool ids come from
  `GET /experimental/tool/ids` (14 ids incl. bash/read/edit/write).
  WARNING: passing all-false breaks chat for mimo/ling/nemotron free models
  (instant empty reply, `finish` null). Gateway therefore leaves tools
  ENABLED and runs sessions in the detected project folder (sandbox fallback) instead.
- Expect large `input` token counts (~20k): opencode prepends its agent
  system prompt. This counts against free-tier quota per request.

## Image parts (verified 2026-09-24, opencode 1.18.32)

`POST /session/{id}/message` accepts file parts alongside text:
`{"type": "file", "mime": "image/png", "url": "file://H:/abs/path.png"}`
(`FilePartInput`: `type`+`mime`+`url` required; plain paths without the
`file://` prefix return `400 BadRequest`). The file must exist on the
serve host; gateway saves inbound base64 images under `sandbox/img/`.
Vision verdict per model: mimo-v2.6-flash-free, muse-spark-1.2/1.3,
space-bunny SEE images; ling-3.0-flash-fin and nemotron-3-ultra/3.5
deny vision in text; mimo-v2.5-free 500s on everything.

## Error / fallback signals

- `4xx/5xx` on session-create or prompt, non-JSON body, missing/empty text
  parts, or `finish` other than `stop` with no text -> treat as failure and
  try the next model in `combo.json`.
- `401` from serve = Basic-auth mismatch (wrong server username/password).
