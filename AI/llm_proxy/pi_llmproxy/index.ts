/**
 * pi-llmproxy — Display the actual model served by the llmproxy proxy in pi's footer.
 *
 * Behavior:
 *  - Static:  GET {proxyUrl}/model/info  ->  alias -> [upstream...] / uuid -> upstream
 *             Footer shows:  🤖 agent-auto → sensenova-6.8-flash-lite
 *  - Dynamic: after_provider_response reads x-llmproxy-upstream (preferred) or the
 *             x-litellm-* compatibility headers
 *             Footer shows:  🤖 served: sensenova-6.8-flash-lite · via llmproxy
 *  - Fallback: proxy unreachable ->  🤖 llmproxy: (proxy unreachable)
 *
 * Config resolution (first hit wins):
 *  1) ./pi_llmproxy/config.json  { "proxyUrl": "...", "pollMs": 30000, "masterKey": "..." }
 *  2) env LLMPROXY_PROXY_URL (overrides file), env LLMPROXY_MASTER_KEY
 *     (LITELLM_* kept as legacy aliases for a while)
 *  3) default http://localhost:4400
 *
 * Usage:  pi -e ./pi_llmproxy/index.ts   (run from the llmproxy project directory)
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

interface LlmproxyConfig {
	proxyUrl?: string;
	pollMs?: number;
	masterKey?: string;
	probe?: boolean;
	probePrompt?: string;
	probeMaxTokens?: number;
}

interface ModelInfoItem {
	model_name?: string;
	litellm_params?: { model?: string };
	model_info?: { id?: string; upstream?: string };
}

type ModelInfoResponse = ModelInfoItem[] | { data?: ModelInfoItem[] };

const STATUS_KEY = "llmproxy-model";
const SERVED_TTL_MS = 5 * 60 * 1000; // dynamic result considered fresh for 5 min

function loadConfig(): LlmproxyConfig {
	const candidates = [
		resolve(process.cwd(), "pi_llmproxy", "config.json"),
		resolve(process.cwd(), "pi-llmproxy", "config.json"),
		resolve(process.cwd(), "config.json"),
	];
	for (const p of candidates) {
		try {
			if (!existsSync(p)) continue;
			const parsed = JSON.parse(readFileSync(p, "utf8"));
			if (parsed && typeof parsed === "object") return parsed as LlmproxyConfig;
		} catch (e) {
			console.error(`[pi-llmproxy] failed to parse ${p}:`, e);
		}
	}
	return {};
}

function stripProviderPrefix(model: string): string {
	// "openai/coding-glm-5.3-free" -> "coding-glm-5.3-free"
	const i = model.indexOf("/");
	return i > 0 && i < model.length - 1 ? model.slice(i + 1) : model;
}

export default function (pi: ExtensionAPI) {
	const cfg = loadConfig();
	const baseUrl = (process.env.LLMPROXY_PROXY_URL
		|| process.env.LITELLM_PROXY_URL
		|| cfg.proxyUrl
		|| "http://127.0.0.1:4400").replace(/\/+$/, "");
	const pollMs = typeof cfg.pollMs === "number" && cfg.pollMs >= 5000 ? cfg.pollMs : 30000;

	let aliasToUpstreams = new Map<string, string[]>();
	let uuidToUpstream = new Map<string, string>();
	let lastServed: { upstream: string; alias?: string; at: number } | null = null;
	let lastError: string | null = null;
	let lastFetchAt = 0;
	let inFlight: Promise<void> | null = null;
	let currentModelId: string | undefined;
	let probeServed: string | null = null;
	let probeAbort: AbortController | null = null;

	function authHeaders(): Record<string, string> {
		const key = process.env.LLMPROXY_MASTER_KEY
			|| process.env.LITELLM_MASTER_KEY
			|| cfg.masterKey;
		return key ? { Authorization: `Bearer ${key}` } : {};
	}

	async function fetchModelInfo(): Promise<void> {
		if (inFlight) return inFlight;
		inFlight = (async () => {
			try {
				const res = await fetch(`${baseUrl}/model/info`, {
					headers: authHeaders(),
					signal: AbortSignal.timeout(5000),
				});
				if (!res.ok) throw new Error(`HTTP ${res.status}`);
				const body = (await res.json()) as ModelInfoResponse;
				// llmproxy returns a bare array; older LiteLLM returns { data: [...] }.
				const items = Array.isArray(body) ? body : (body.data ?? []);
				const a2u = new Map<string, string[]>();
				const u2u = new Map<string, string>();
				for (const item of items) {
					const alias = item.model_name;
					// Prefer model_info.upstream (llmproxy convention: the upstream name);
					// fall back to litellm_params.model (LiteLLM convention: real model id).
					const upstream = item.model_info?.upstream || item.litellm_params?.model;
					if (!alias || !upstream) continue;
					const list = a2u.get(alias) ?? [];
					if (!list.includes(upstream)) list.push(upstream);
					a2u.set(alias, list);
					const id = item.model_info?.id;
					if (id) u2u.set(id, upstream);
				}
				aliasToUpstreams = a2u;
				uuidToUpstream = u2u;
				lastError = null;
				lastFetchAt = Date.now();
			} catch (e) {
				lastError = e instanceof Error ? e.message : String(e);
			} finally {
				inFlight = null;
			}
		})();
		return inFlight;
	}

	async function probeServedModel(alias: string): Promise<void> {
		if (!cfg.probe) return;
		probeAbort?.abort();
		probeAbort = new AbortController();
		try {
			const res = await fetch(`${baseUrl}/v1/chat/completions`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					...authHeaders(),
				},
				body: JSON.stringify({
					model: alias,
					messages: [{ role: "user", content: cfg.probePrompt ?? "ping" }],
					max_tokens: cfg.probeMaxTokens ?? 3,
				}),
				signal: AbortSignal.timeout(30_000),
			});
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const headers: Record<string, string> = {};
			res.headers.forEach((v, k) => { headers[k.toLowerCase()] = v; });
			// llmproxy-native headers first, litellm compatibility headers as fallback.
			const uuid = headers["x-llmproxy-model-id"] || headers["x-litellm-model-id"];
			const group = headers["x-llmproxy-model-group"] || headers["x-litellm-model-group"];
			const upstreamFromHeader = headers["x-llmproxy-upstream"];
			const modelFromHeader = headers["x-llmproxy-model"];
			let upstream: string | undefined;
			// Prefer llmproxy's explicit x-llmproxy-model (real model id); fall back to
			// resolving the deployment id via /model/info; then alias -> first candidate.
			if (modelFromHeader) {
				upstream = modelFromHeader;
			} else if (upstreamFromHeader && aliasToUpstreams.has(group ?? "")) {
				// header says upstream name; map to real model id via /model/info
				const byUpstream = [...aliasToUpstreams.entries()].find(
					([a, ups]) => a === group && ups.includes(upstreamFromHeader),
				);
				upstream = byUpstream?.[1][0];
			} else if (uuid && uuidToUpstream.has(uuid)) {
				upstream = uuidToUpstream.get(uuid);
			} else if (group && aliasToUpstreams.has(group)) {
				upstream = aliasToUpstreams.get(group)![0];
			}
			if (upstream) {
				lastServed = { upstream, alias: group, at: Date.now() };
				currentModelId = group ?? alias;
				lastError = null;
				probeServed = upstream;
			}
		} catch (e) {
			lastError = e instanceof Error ? e.message : String(e);
		} finally {
			probeAbort = null;
		}
	}

	function resolveStaticUpstream(alias: string): string | null {
		const list = aliasToUpstreams.get(alias);
		if (list && list.length > 0) return list[0]; // simple-shuffle: show first candidate
		const all = new Set<string>();
		for (const ups of aliasToUpstreams.values()) for (const u of ups) all.add(u);
		return all.has(alias) ? alias : null; // direct alias (pi model id == upstream)
	}

	function paint(theme: any, color: string, text: string): string {
		try {
			return theme?.fg ? theme.fg(color, text) : text;
		} catch {
			return text;
		}
	}

	function buildStatus(theme: any): string {
		const bot = paint(theme, "accent", "🤖");
		if (lastServed && Date.now() - lastServed.at < SERVED_TTL_MS) {
			const served = paint(theme, "success", `served: ${stripProviderPrefix(lastServed.upstream)}`);
			const dimmed = paint(theme, "dim", ` · via llmproxy`);
			return `${bot} ${served}${dimmed}`;
		}
		if (probeServed) {
			const served = paint(theme, "success", `served: ${stripProviderPrefix(probeServed)}`);
			const dimmed = paint(theme, "dim", ` · via llmproxy (probed)`);
			return `${bot} ${served}${dimmed}`;
		}
		if (currentModelId) {
			const upstream = resolveStaticUpstream(currentModelId);
			if (upstream) {
				const alias = paint(theme, "fg", currentModelId);
				const arrow = paint(theme, "dim", "→");
				const up = paint(theme, "success", stripProviderPrefix(upstream));
				return `${bot} ${alias} ${arrow} ${up}`;
			}
		}
		if (lastError) {
			return `${bot} ${paint(theme, "warning", `llmproxy: (${lastError})`)}`;
		}
		if (aliasToUpstreams.size > 0) {
			return `${bot} ${paint(theme, "dim", "llmproxy: model unknown")}`;
		}
		return `${bot} ${paint(theme, "dim", "llmproxy: loading...")}`;
	}

	function refreshStatus(ctx: any): void {
		try {
			ctx?.ui?.setStatus?.(STATUS_KEY, buildStatus(ctx?.ui?.theme));
		} catch {
			/* never break the session over a status line */
		}
	}

	pi.on("session_start", async (_event, ctx) => {
		await fetchModelInfo();
		const modelId = ctx.model?.id;
		if (modelId) await probeServedModel(modelId);
		refreshStatus(ctx);
	});

	pi.on("model_select", async (event, ctx) => {
		const newId = event.model?.id ?? currentModelId;
		if (newId && newId !== currentModelId) currentModelId = newId;
		probeServed = null;
		lastServed = null;
		if (Date.now() - lastFetchAt > pollMs - 1000) await fetchModelInfo();
		if (currentModelId) await probeServedModel(currentModelId);
		refreshStatus(ctx);
	});

	pi.on("agent_start", async (_event, ctx) => {
		if (Date.now() - lastFetchAt > pollMs - 1000) await fetchModelInfo();
		refreshStatus(ctx);
	});

	pi.on("after_provider_response", async (event, ctx) => {
		if (event.status && event.status >= 200 && event.status < 300) {
			const headers = event.headers ?? {};
			const lower: Record<string, string> = {};
			for (const k of Object.keys(headers)) lower[k.toLowerCase()] = String(headers[k]);

			const uuid = lower["x-llmproxy-model-id"] || lower["x-litellm-model-id"];
			const group = lower["x-llmproxy-model-group"] || lower["x-litellm-model-group"];
			const upstreamFromHeader = lower["x-llmproxy-upstream"];
			const modelFromHeader = lower["x-llmproxy-model"];

			let upstream: string | undefined;
			if (modelFromHeader) {
				upstream = modelFromHeader;
			} else if (upstreamFromHeader && group && aliasToUpstreams.has(group)) {
				upstream = aliasToUpstreams.get(group)!.includes(upstreamFromHeader)
					? upstreamFromHeader
					: aliasToUpstreams.get(group)![0];
			} else if (uuid && uuidToUpstream.has(uuid)) {
				upstream = uuidToUpstream.get(uuid);
			} else if (group && aliasToUpstreams.has(group)) {
				upstream = aliasToUpstreams.get(group)![0];
			}
			if (upstream) {
				lastServed = { upstream, alias: group, at: Date.now() };
				currentModelId = group ?? currentModelId;
			}
		}
		refreshStatus(ctx);
	});
}
