export interface TokenCandidate {
	token_id: number;
	text: string;
	probability: number;
	logit: number;
	rank: number;
}

export interface HistoryEntry {
	token_id: number;
	text: string;
	probability: number;
	rank: number;
}

export interface Distribution {
	tokens: TokenCandidate[];
	sequence_length: number;
	vocab_size: number;
	prefill_ms?: number;
	prefill_tps?: number;
	step_ms?: number;
}

export interface ModelInfo {
	model_path: string;
	total_parameters: number;
	bits_per_weight: number;
	vocab_size: number;
}

export interface SessionResponse {
	distribution: Distribution;
	history: HistoryEntry[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, {
		headers: { "Content-Type": "application/json" },
		...init,
	});
	if (!res.ok) {
		const msg = await res.text().catch(() => res.statusText);
		throw new Error(msg);
	}
	return res.json();
}

export const api = {
	health: () => request<{ status: string }>("/api/health"),

	model: () => request<ModelInfo>("/api/model"),

	init: (prompt: string, temperature: number, top_k: number) =>
		request<SessionResponse>("/api/init", {
			method: "POST",
			body: JSON.stringify({ prompt, temperature, top_k }),
		}),

	step: (token_id: number, probability: number, rank: number, temperature: number, top_k: number) =>
		request<SessionResponse>("/api/step", {
			method: "POST",
			body: JSON.stringify({ token_id, probability, rank, temperature, top_k }),
		}),

	reset: () => request<{ status: string }>("/api/reset", { method: "POST" }),
};
