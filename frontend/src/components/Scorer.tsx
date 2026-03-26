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
	const confidence = result ? Math.exp(result.avg_log_prob) * 100 : 0;
	const interp = interpret(confidence);

	return (
		<div className="h-full overflow-y-auto">
			<div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
				<section className="grid grid-cols-[1fr_280px] gap-5">
					<div className="space-y-3">
						<label className="block">
							<span className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">User Message</span>
							<textarea
								value={userMessage}
								onChange={(e) => setUserMessage(e.target.value)}
								placeholder="What would the user say?"
								rows={3}
								className="w-full bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-4 py-3 text-sm font-mono resize-y focus:outline-none focus:border-zinc-600 placeholder:text-zinc-700"
							/>
						</label>
						<label className="block">
							<span className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">Assistant Reply</span>
							<textarea
								value={assistantReply}
								onChange={(e) => setAssistantReply(e.target.value)}
								placeholder="What reply do you want to score?"
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
							{loading ? "Analyzing…" : "Score Likelihood"}
						</button>
						{error && <p className="text-red-400 text-xs mt-1">{error}</p>}
					</div>

					<div className="space-y-4">
						<label className="block">
							<span className="block text-[10px] uppercase tracking-wider text-zinc-500 mb-1.5">System Prompt</span>
							<textarea
								value={systemPrompt}
								onChange={(e) => setSystemPrompt(e.target.value)}
								placeholder="Optional model instructions…"
								rows={4}
								className="w-full bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-3 py-2.5 text-xs font-mono resize-y focus:outline-none focus:border-zinc-600 placeholder:text-zinc-700"
							/>
						</label>
						{modelInfo && (
							<div className="text-[10px] text-zinc-600 space-y-0.5">
								<p className="font-mono text-zinc-500">{modelInfo.model_path.split("/").pop()}</p>
								<p>
									{(modelInfo.total_parameters / 1e9).toFixed(1)}B params · {modelInfo.bits_per_weight} bits ·{" "}
									{modelInfo.vocab_size.toLocaleString()} vocab
								</p>
							</div>
						)}
					</div>
				</section>

				{loading && (
					<div className="text-center py-12 text-zinc-500 text-sm animate-pulse">
						Analyzing how the model would generate this reply…
					</div>
				)}

				{result && (
					<>
						<section className="flex items-center gap-6 bg-zinc-900/40 border border-zinc-800/40 rounded-xl p-6">
							<ConfidenceRing value={confidence} />
							<div className="flex-1 space-y-2">
								<div>
									<p className="text-lg font-semibold">
										<span className={interp.color}>{interp.label}</span>
									</p>
									<p className="text-sm text-zinc-400">{interp.description}</p>
								</div>
								<div className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
									<Stat label="Perplexity" value={result.perplexity.toFixed(2)} hint="Lower = more expected" />
									<Stat label="Avg Log-Prob" value={result.avg_log_prob.toFixed(3)} hint="Per token" />
									<Stat label="Total Log-Prob" value={result.total_log_prob.toFixed(3)} hint="All tokens" />
									<Stat label="Reply Tokens" value={String(result.reply_length)} />
									<Stat label="Prompt Tokens" value={String(result.prompt_length)} />
									<Stat label="Inference" value={`${result.elapsed_ms.toFixed(0)}ms`} />
								</div>
							</div>
						</section>

						<section className="space-y-2">
							<div className="flex items-center justify-between">
								<h2 className="text-xs font-medium text-zinc-400">Token-by-Token Breakdown</h2>
								<div className="flex items-center gap-1.5 text-[10px] text-zinc-600">
									<div className="flex h-2 rounded-full overflow-hidden w-24">
										{[0, 1, 2, 3, 4].map((i) => (
											<div key={i} className="flex-1" style={{ backgroundColor: probBg(10 ** -(4 - i)) }} />
										))}
									</div>
									<span>unlikely → confident</span>
								</div>
							</div>
							<TokenHeatmap tokens={result.tokens} selected={selected} onSelect={setSelected} />
						</section>
					</>
				)}
			</div>
		</div>
	);
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
	return (
		<span className="text-zinc-500" title={hint}>
			{label} <span className="font-mono text-zinc-300">{value}</span>
		</span>
	);
}

function ConfidenceRing({ value }: { value: number }) {
	const r = 42;
	const c = 2 * Math.PI * r;
	const offset = c * (1 - Math.min(value, 100) / 100);
	const color = value >= 60 ? "#22c55e" : value >= 30 ? "#eab308" : value >= 10 ? "#f97316" : "#ef4444";

	return (
		<svg
			width="100"
			height="100"
			viewBox="0 0 100 100"
			className="shrink-0"
			aria-label={`${Math.round(value)}% confidence`}
		>
			<circle cx="50" cy="50" r={r} fill="none" stroke="currentColor" className="text-zinc-800/60" strokeWidth="6" />
			<circle
				cx="50"
				cy="50"
				r={r}
				fill="none"
				stroke={color}
				strokeWidth="6"
				strokeDasharray={c}
				strokeDashoffset={offset}
				strokeLinecap="round"
				transform="rotate(-90 50 50)"
				className="transition-all duration-700"
			/>
			<text x="50" y="44" textAnchor="middle" dominantBaseline="central" fill="white" fontSize="22" fontWeight="bold">
				{Math.round(value)}
			</text>
			<text x="50" y="63" textAnchor="middle" fill="#a1a1aa" fontSize="10">
				%
			</text>
		</svg>
	);
}

function interpret(v: number): { label: string; color: string; description: string } {
	if (v >= 60)
		return {
			label: "Highly Natural",
			color: "text-green-400",
			description: "The model would very likely produce this exact reply.",
		};
	if (v >= 35)
		return { label: "Natural", color: "text-emerald-400", description: "The model finds this reply quite expected." };
	if (v >= 15)
		return {
			label: "Plausible",
			color: "text-yellow-400",
			description: "A reasonable reply, but not the model's first choice.",
		};
	if (v >= 5)
		return {
			label: "Unusual",
			color: "text-orange-400",
			description: "The model would not typically generate this reply.",
		};
	return { label: "Very Unlikely", color: "text-red-400", description: "The model finds this reply very unexpected." };
}

function probBg(p: number): string {
	const log = Math.log10(Math.max(p, 1e-6));
	const t = Math.min(1, Math.max(0, (log + 4) / 4));
	const hue = t * 120;
	return `hsla(${hue}, 75%, 45%, 0.35)`;
}
