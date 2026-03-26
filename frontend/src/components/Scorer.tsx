"use client";

import { useEffect, useRef, useState } from "react";
import { api, type ModelInfo, type ScoreResult } from "@/lib/api";
import TokenHeatmap from "./TokenHeatmap";

export default function Scorer() {
	const [systemPrompt, setSystemPrompt] = useState("");
	const [userMessage, setUserMessage] = useState("");
	const [assistantReply, setAssistantReply] = useState("");
	const [result, setResult] = useState<ScoreResult | null>(null);
	const [selected, setSelected] = useState<number | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
	const abortRef = useRef<AbortController | null>(null);

	useEffect(() => {
		api
			.model()
			.then(setModelInfo)
			.catch(() => {});
	}, []);

	async function handleScore() {
		if (!userMessage.trim() || !assistantReply.trim()) return;

		abortRef.current?.abort();
		const controller = new AbortController();
		abortRef.current = controller;

		setLoading(true);
		setError(null);
		setResult(null);
		setSelected(null);

		try {
			const res = await api.score(userMessage, assistantReply, systemPrompt || null, controller.signal);
			if (!controller.signal.aborted) setResult(res);
		} catch (e) {
			if (controller.signal.aborted) return;
			setError(e instanceof Error ? e.message : "Scoring failed");
		} finally {
			if (!controller.signal.aborted) setLoading(false);
		}
	}

	const canScore = userMessage.trim().length > 0 && assistantReply.trim().length > 0 && !loading;

	return (
		<div className="h-full flex">
			<main className="flex-1 flex flex-col min-h-0">
				<section className="shrink-0 border-b border-zinc-800/60 p-5 space-y-3">
					<label className="block">
						<span className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">User Message</span>
						<textarea
							value={userMessage}
							onChange={(e) => setUserMessage(e.target.value)}
							placeholder="What is the capital of France?"
							rows={3}
							className="w-full bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-4 py-3 text-sm font-mono resize-y focus:outline-none focus:border-zinc-600 placeholder:text-zinc-700"
						/>
					</label>
					<label className="block">
						<span className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">Assistant Reply</span>
						<textarea
							value={assistantReply}
							onChange={(e) => setAssistantReply(e.target.value)}
							placeholder="The capital of France is Paris."
							rows={3}
							className="w-full bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-4 py-3 text-sm font-mono resize-y focus:outline-none focus:border-zinc-600 placeholder:text-zinc-700"
						/>
					</label>
					<button
						type="button"
						onClick={handleScore}
						disabled={!canScore}
						className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors bg-violet-600/80 hover:bg-violet-500/80 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed"
					>
						{loading ? "Scoring…" : "Score Likelihood"}
					</button>
					{error && <p className="text-red-400 text-xs">{error}</p>}
				</section>

				<section className="flex-1 min-h-0 overflow-hidden">
					{result?.tokens.length ? (
						<TokenHeatmap tokens={result.tokens} selected={selected} onSelect={setSelected} />
					) : (
						<div className="h-full flex items-center justify-center text-zinc-600 text-sm">
							{loading ? "Computing likelihood scores…" : "Provide a user message and assistant reply to score"}
						</div>
					)}
				</section>
			</main>

			<aside className="w-72 shrink-0 border-l border-zinc-800/60 overflow-y-auto">
				<div className="p-4 space-y-5">
					<label className="block">
						<span className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">System Prompt</span>
						<textarea
							value={systemPrompt}
							onChange={(e) => setSystemPrompt(e.target.value)}
							placeholder="Optional system prompt…"
							rows={4}
							className="w-full bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-3 py-2.5 text-xs font-mono resize-y focus:outline-none focus:border-zinc-600 placeholder:text-zinc-700"
						/>
					</label>

					{result && (
						<div>
							<h3 className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2.5">Metrics</h3>
							<div className="space-y-2">
								<Metric label="Perplexity" value={result.perplexity.toFixed(2)} />
								<Metric label="Avg Log-Prob" value={result.avg_log_prob.toFixed(4)} />
								<Metric label="Total Log-Prob" value={result.total_log_prob.toFixed(4)} />
								<Metric label="Reply Tokens" value={String(result.reply_length)} />
								<Metric label="Prompt Tokens" value={String(result.prompt_length)} />
								<Metric label="Inference" value={`${result.elapsed_ms.toFixed(0)} ms`} />
							</div>
						</div>
					)}

					{modelInfo && (
						<div>
							<h3 className="text-[10px] uppercase tracking-wider text-zinc-500 mb-2.5">Model</h3>
							<p className="text-xs text-zinc-400 font-mono break-all leading-relaxed">{modelInfo.model_path}</p>
							<div className="mt-1.5 text-[10px] text-zinc-600 space-y-0.5">
								<p>{(modelInfo.total_parameters / 1e9).toFixed(1)}B params</p>
								<p>{modelInfo.bits_per_weight} bits/weight</p>
								<p>{modelInfo.vocab_size.toLocaleString()} vocab</p>
							</div>
						</div>
					)}
				</div>
			</aside>
		</div>
	);
}

function Metric({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-center justify-between text-xs">
			<span className="text-zinc-500">{label}</span>
			<span className="font-mono text-zinc-300">{value}</span>
		</div>
	);
}
