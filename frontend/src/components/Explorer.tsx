"use client";

import { useEffect, useRef, useState } from "react";
import { api, type Distribution, type ModelInfo, type TokenCandidate } from "@/lib/api";
import ParamsPanel from "./ParamsPanel";
import TokenList from "./TokenList";

export default function Explorer() {
	const [text, setText] = useState("");
	const [systemPrompt, setSystemPrompt] = useState("");
	const [distribution, setDistribution] = useState<Distribution | null>(null);
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
	const [temperature, setTemperature] = useState(1.0);
	const [topK, setTopK] = useState(200);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
	const requestIdRef = useRef(0);
	const textRef = useRef(text);
	const sysRef = useRef(systemPrompt);
	textRef.current = text;
	sysRef.current = systemPrompt;

	useEffect(() => {
		api
			.model()
			.then(setModelInfo)
			.catch(() => {});
	}, []);

	async function predict(t: string, sp: string, temp: number, tk: number) {
		if (!t.trim()) {
			setDistribution(null);
			return;
		}
		const id = ++requestIdRef.current;
		setLoading(true);
		setError(null);
		try {
			const res = await api.predict(t, sp || null, temp, tk);
			if (requestIdRef.current !== id) return;
			setDistribution(res.distribution);
		} catch (e) {
			if (requestIdRef.current !== id) return;
			setError(e instanceof Error ? e.message : "Prediction failed");
		} finally {
			if (requestIdRef.current === id) setLoading(false);
		}
	}

	function schedulePrediction(t: string, sp: string, temp: number, tk: number, delay = 400) {
		clearTimeout(debounceRef.current);
		debounceRef.current = setTimeout(() => predict(t, sp, temp, tk), delay);
	}

	function handleTextChange(newText: string) {
		setText(newText);
		schedulePrediction(newText, systemPrompt, temperature, topK);
		autoResize();
	}

	function handleSystemPromptChange(newPrompt: string) {
		setSystemPrompt(newPrompt);
		if (text.trim()) schedulePrediction(text, newPrompt, temperature, topK);
	}

	function handleTemperatureChange(v: number) {
		setTemperature(v);
		if (textRef.current.trim()) schedulePrediction(textRef.current, sysRef.current, v, topK, 150);
	}

	function handleTopKChange(v: number) {
		setTopK(v);
		if (textRef.current.trim()) schedulePrediction(textRef.current, sysRef.current, temperature, v, 150);
	}

	function selectToken(token: TokenCandidate) {
		const newText = text + token.text;
		setText(newText);
		clearTimeout(debounceRef.current);
		predict(newText, systemPrompt, temperature, topK);
		requestAnimationFrame(() => {
			const el = textareaRef.current;
			if (el) {
				el.focus();
				el.selectionStart = newText.length;
				el.selectionEnd = newText.length;
				autoResize();
			}
		});
	}

	function autoResize() {
		requestAnimationFrame(() => {
			const el = textareaRef.current;
			if (el) {
				el.style.height = "auto";
				el.style.height = `${Math.min(el.scrollHeight, 320)}px`;
			}
		});
	}

	useEffect(() => {
		if (!distribution || loading) return;
		function onKey(e: KeyboardEvent) {
			if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
			const tokens = distribution?.tokens;
			if (!tokens?.length) return;
			if (e.key === "Enter") {
				e.preventDefault();
				selectToken(tokens[0]);
				return;
			}
			const n = Number.parseInt(e.key, 10);
			if (n >= 1 && n <= 9 && tokens[n - 1]) {
				e.preventDefault();
				selectToken(tokens[n - 1]);
			}
		}
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	});

	function handleKeyDown(e: React.KeyboardEvent) {
		if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
			e.preventDefault();
			clearTimeout(debounceRef.current);
			predict(text, systemPrompt, temperature, topK);
		}
	}

	return (
		<div className="h-screen flex flex-col">
			<header className="shrink-0 border-b border-zinc-800/60 px-6 py-3.5 flex items-center justify-between">
				<div>
					<h1 className="text-base font-semibold tracking-tight">Token Explorer</h1>
					<p className="text-[11px] text-zinc-500 mt-0.5">Interactive Next Token Prediction</p>
				</div>
				{systemPrompt ? (
					<span className="text-[10px] text-emerald-500/80 border border-emerald-800/40 px-2 py-0.5 rounded">
						Chat Template Active
					</span>
				) : (
					<span className="text-[10px] text-zinc-600 border border-zinc-800/40 px-2 py-0.5 rounded">
						Raw Completion
					</span>
				)}
			</header>

			<div className="flex flex-1 min-h-0">
				<main className="flex-1 flex flex-col min-h-0">
					<section className="shrink-0 border-b border-zinc-800/60 p-5">
						<textarea
							ref={textareaRef}
							value={text}
							onChange={(e) => handleTextChange(e.target.value)}
							onKeyDown={handleKeyDown}
							placeholder="Start typing to see next-token predictions…"
							rows={4}
							className="w-full bg-zinc-900/70 border border-zinc-800 rounded-lg px-4 py-3 text-sm font-mono leading-relaxed focus:outline-none focus:ring-1 focus:ring-violet-500/40 placeholder:text-zinc-600"
							style={{ resize: "none", overflow: "auto", minHeight: "6rem", maxHeight: "20rem" }}
						/>
						{error && <p className="text-red-400 text-xs mt-2">{error}</p>}
						<div className="flex items-center gap-3 mt-2 text-[10px] text-zinc-600">
							<span>{text.length} chars</span>
							{distribution?.sequence_length != null && (
								<>
									<span>·</span>
									<span>{distribution.sequence_length} tokens</span>
								</>
							)}
							{loading && <span className="text-violet-400 animate-pulse">Predicting…</span>}
							<span className="ml-auto opacity-50">Cmd+Enter to force predict</span>
						</div>
					</section>

					<section className="flex-1 min-h-0">
						{distribution?.tokens.length ? (
							<TokenList tokens={distribution.tokens} onSelect={selectToken} disabled={loading} />
						) : (
							<div className="h-full flex items-center justify-center text-zinc-600 text-sm">
								{loading ? "Processing…" : "Type something to see predictions"}
							</div>
						)}
					</section>
				</main>

				<aside className="w-72 shrink-0 border-l border-zinc-800/60 overflow-y-auto">
					<ParamsPanel
						temperature={temperature}
						topK={topK}
						onTemperatureChange={handleTemperatureChange}
						onTopKChange={handleTopKChange}
						systemPrompt={systemPrompt}
						onSystemPromptChange={handleSystemPromptChange}
						modelInfo={modelInfo}
						distribution={distribution}
					/>
				</aside>
			</div>
		</div>
	);
}
