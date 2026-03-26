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

export interface TokenScore {
	id: number;
	text: string;
	probability: number;
	log_prob: number;
	rank: number;
	alternatives: { id: number; text: string; probability: number }[];
}

export interface ScoreResult {
	tokens: TokenScore[];
	total_log_prob: number;
	avg_log_prob: number;
	perplexity: number;
	prompt_length: number;
	reply_length: number;
	elapsed_ms: number;
}

export class SupersededError extends Error {
	constructor() {
		super("Superseded");
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, {
		headers: { "Content-Type": "application/json" },
		...init,
	});
	if (res.status === 409) throw new SupersededError();
	if (!res.ok) {
		const msg = await res.text().catch(() => res.statusText);
		throw new Error(msg);
	}
	return res.json();
}

export const api = {
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

	score: (
		userMessage: string,
		assistantReply: string,
		systemPrompt: string | null,
		topK: number,
		signal?: AbortSignal,
	) =>
		request<ScoreResult>("/api/score", {
			method: "POST",
			body: JSON.stringify({
				user_message: userMessage,
				assistant_reply: assistantReply,
				system_prompt: systemPrompt,
				top_k: topK,
			}),
			signal,
		}),

	reset: () => request<{ status: string }>("/api/reset", { method: "POST" }),
};
