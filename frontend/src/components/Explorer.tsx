"use client";

import { useEffect, useRef, useState } from "react";
import { api, type Distribution, type ModelInfo, type TokenCandidate, type TokenSpan } from "@/lib/api";
import ParamsPanel from "./ParamsPanel";
import TokenEditor, { charOffsetToTokenCursor, type TokenEditorHandle, tokenCursorToCharOffset } from "./TokenEditor";
import TokenList from "./TokenList";

export default function Explorer() {
	const [text, setText] = useState("");
	const [tokens, setTokens] = useState<TokenSpan[]>([]);
	const [charOffset, setCharOffset] = useState(0);
	const [distribution, setDistribution] = useState<Distribution | null>(null);
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
	const [systemPrompt, setSystemPrompt] = useState("");
	const [temperature, setTemperature] = useState(1.0);
	const [topK, setTopK] = useState(200);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const editorRef = useRef<TokenEditorHandle>(null);
	const predictDebounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
	const predictIdRef = useRef(0);
	const tokenizeIdRef = useRef(0);
	const textRef = useRef(text);
	const sysRef = useRef(systemPrompt);
	textRef.current = text;
	sysRef.current = systemPrompt;

	const cursor = charOffsetToTokenCursor(tokens, charOffset);

	useEffect(() => {
		api
			.model()
			.then(setModelInfo)
			.catch(() => {});
	}, []);

	async function doTokenize(t: string) {
		if (!t) {
			setTokens([]);
			return;
		}
		const id = ++tokenizeIdRef.current;
		try {
			const res = await api.tokenize(t);
			if (tokenizeIdRef.current === id) setTokens(res.tokens);
		} catch {
			/* non-critical */
		}
	}

	async function doPredict(t: string, sp: string, temp: number, tk: number) {
		if (!t.trim()) {
			setDistribution(null);
			return;
		}
		const id = ++predictIdRef.current;
		setLoading(true);
		setError(null);
		try {
			const res = await api.predict(t, sp || null, temp, tk);
			if (predictIdRef.current !== id) return;
			setDistribution(res.distribution);
		} catch (e) {
			if (predictIdRef.current !== id) return;
			setError(e instanceof Error ? e.message : "Prediction failed");
		} finally {
			if (predictIdRef.current === id) setLoading(false);
		}
	}

	function schedulePrediction(t: string, sp: string, temp: number, tk: number, delay = 400) {
		clearTimeout(predictDebounceRef.current);
		predictDebounceRef.current = setTimeout(() => doPredict(t, sp, temp, tk), delay);
	}

	function handleTextChange(newText: string, newCharOffset: number) {
		setText(newText);
		setCharOffset(newCharOffset);
		doTokenize(newText);
		schedulePrediction(newText, systemPrompt, temperature, topK);
	}

	function handleCursorChange(tokenPos: number) {
		setCharOffset(tokenCursorToCharOffset(tokens, tokenPos));
	}

	function selectPredictedToken(token: TokenCandidate) {
		const newText = text + token.text;
		setText(newText);
		setCharOffset(newText.length);
		doTokenize(newText);
		clearTimeout(predictDebounceRef.current);
		doPredict(newText, systemPrompt, temperature, topK);
		requestAnimationFrame(() => editorRef.current?.focus());
	}

	function handleSystemPromptChange(sp: string) {
		setSystemPrompt(sp);
		if (text.trim()) schedulePrediction(text, sp, temperature, topK);
	}

	function handleTemperatureChange(v: number) {
		setTemperature(v);
		if (textRef.current.trim()) schedulePrediction(textRef.current, sysRef.current, v, topK, 150);
	}

	function handleTopKChange(v: number) {
		setTopK(v);
		if (textRef.current.trim()) schedulePrediction(textRef.current, sysRef.current, temperature, v, 150);
	}

	function forcePredict() {
		clearTimeout(predictDebounceRef.current);
		doPredict(text, systemPrompt, temperature, topK);
	}

	useEffect(() => {
		if (!distribution || loading) return;
		function onKey(e: KeyboardEvent) {
			if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
			const t = distribution?.tokens;
			if (!t?.length) return;
			if (e.key === "Enter") {
				e.preventDefault();
				selectPredictedToken(t[0]);
				return;
			}
			const n = Number.parseInt(e.key, 10);
			if (n >= 1 && n <= 9 && t[n - 1]) {
				e.preventDefault();
				selectPredictedToken(t[n - 1]);
			}
		}
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	});

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
						<TokenEditor
							ref={editorRef}
							text={text}
							tokens={tokens}
							cursor={cursor}
							onTextChange={handleTextChange}
							onCursorChange={handleCursorChange}
							onForcePredict={forcePredict}
						/>
						{error && <p className="text-red-400 text-xs mt-2">{error}</p>}
						<div className="flex items-center gap-3 mt-2 text-[10px] text-zinc-600">
							<span>{tokens.length} tokens</span>
							<span>·</span>
							<span>{text.length} chars</span>
							{loading && (
								<>
									<span>·</span>
									<span className="text-violet-400 animate-pulse">Predicting…</span>
								</>
							)}
							<span className="ml-auto opacity-50">←→ nav · ⌫ del token · ⌘⌫ clear · ⌘↵ predict</span>
						</div>
					</section>

					<section className="flex-1 min-h-0">
						{distribution?.tokens.length ? (
							<TokenList tokens={distribution.tokens} onSelect={selectPredictedToken} disabled={loading} />
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
