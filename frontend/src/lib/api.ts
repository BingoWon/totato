export interface TokenCandidate {
	id: number;
	text: string;
	probability: number;
	logit: number;
	rank: number;
}

export interface TokenSpan {
	id: number;
	text: string;
}

export interface Distribution {
	tokens: TokenCandidate[];
	sequence_length: number;
	vocab_size: number;
	prefill_ms?: number;
	prefill_tps?: number;
	step_ms?: number;
	cached?: boolean;
}

export interface ModelInfo {
	model_path: string;
	total_parameters: number;
	bits_per_weight: number;
	vocab_size: number;
	has_chat_template: boolean;
}

export interface PredictResponse {
	distribution: Distribution;
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

	tokenize: (text: string, signal?: AbortSignal) =>
		request<{ tokens: TokenSpan[] }>("/api/tokenize", {
			method: "POST",
			body: JSON.stringify({ text }),
			signal,
		}),

	predict: (text: string, systemPrompt: string | null, temperature: number, topK: number, signal?: AbortSignal) =>
		request<PredictResponse>("/api/predict", {
			method: "POST",
			body: JSON.stringify({ text, system_prompt: systemPrompt, temperature, top_k: topK }),
			signal,
		}),

	reset: () => request<{ status: string }>("/api/reset", { method: "POST" }),
};
