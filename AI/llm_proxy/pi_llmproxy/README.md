# pi-llmproxy

Pi agent extension that resolves the currently active model through the
**llmproxy** proxy and displays the actually-served upstream model in the TUI
footer.

## What it shows

- **Probed (on startup / model switch):**
  `🤖 served: sensenova-6.8-flash-lite · via llmproxy (probed)`
  A lightweight `max_tokens:3` completion is sent at session start and on each
  model switch. The real model is read from `x-llmproxy-model` (or, as a
  fallback, from the LiteLLM-compatible `x-litellm-model-id` + `/model/info`).
- **Dynamic (after a chat completion):**
  `🤖 served: sensenova-6.8-flash-lite · via llmproxy`
  The real response header from an actual agent message (5 min TTL).
- **Static (proxy reachable, no recent probe/traffic):**
  `🤖 agent-auto → sensenova-6.8-flash-lite`
  First candidate from `GET /model/info`.
- **Proxy unreachable:**
  `🤖 llmproxy: (HTTP 503)` / `🤖 llmproxy: (fetch failed)`

## Install / Run

From the llmproxy project root:

```bash
pi -e ./pi_llmproxy/index.ts
```

No npm install — pi loads the TypeScript file directly.

## Configuration

`pi_llmproxy/config.json`:

```json
{
  "proxyUrl": "http://127.0.0.1:4400",
  "pollMs": 30000,
  "probe": true,
  "probePrompt": "ping",
  "probeMaxTokens": 3,
  "masterKey": ""
}
```

Config file resolution (first hit wins):

1. `./pi_llmproxy/config.json`
2. `./pi-llmproxy/config.json` (legacy path)
3. `./config.json`

Environment variable overrides:

| Key | Default | Description |
|-----|---------|-------------|
| `proxyUrl` | `http://127.0.0.1:4400` | llmproxy proxy base URL. Env override: `LLMPROXY_PROXY_URL` (legacy `LITELLM_PROXY_URL` still works). |
| `pollMs` | `30000` | How often `/model/info` cache refreshes (min 5 000 ms). |
| `probe` | `true` | Send a minimal completion at startup / on model switch to learn the real upstream. |
| `probePrompt` | `"ping"` | Content of the probe completion request. |
| `probeMaxTokens` | `3` | Max output tokens for the probe request. Must be `> 2`: some providers (e.g. bai/GLM) reject `max_tokens <= 2` with a 400. |
| `masterKey` | `""` | Bearer token for protected proxies. Env override: `LLMPROXY_MASTER_KEY` (legacy `LITELLM_MASTER_KEY` still works). |

## How "actual model" is determined

llmproxy adds the following headers to every chat completion response:

- `x-llmproxy-model` — the **real model id** actually served (used directly).
- `x-llmproxy-upstream` — the upstream name (e.g. `sensenova`).
- `x-llmproxy-model-id` — deployment id `{virtual}@{upstream}`, same as
  `/model/info`'s `model_info.id`.
- `x-litellm-model-id` / `x-litellm-model-group` — compatibility aliases of
  the two above, kept for older extensions.

Resolution order in the extension:

1. `x-llmproxy-model` header (authoritative, real model id).
2. `x-llmproxy-upstream` + `/model/info` `model_info.upstream` mapping.
3. `x-llmproxy-model-id` (or `x-litellm-model-id`) → `/model/info` `model_info.id`
   → `litellm_params.model`.
4. Alias → first candidate in `/model/info` (static fallback).

`GET /model/info` is expected to return either a bare JSON array (llmproxy
convention) or `{ "data": [...] }` (LiteLLM convention); both are supported.

## Files

- `index.ts` — extension entry (loaded by `pi -e`).
- `config.json` — proxy URL, polling interval, probe settings.
- `README.md` — this file.
