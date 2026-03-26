"use client";

import { useEffect, useRef, useState } from "react";
import {
	api,
	type Distribution,
	type ModelInfo,
	SupersededError,
	type TokenCandidate,
	type TokenSpan,
} from "@/lib/api";
import ParamsPanel from "./ParamsPanel";
import TokenEditor, { type TokenEditorHandle } from "./TokenEditor";
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
	const tokenizeDebounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
	const predictDebounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);
	const tokenizeAbortRef = useRef<AbortController | null>(null);
	const predictAbortRef = useRef<AbortController | null>(null);
	const stateRef = useRef({ text, systemPrompt, temperature, topK });
	stateRef.current = { text, systemPrompt, temperature, topK };

	useEffect(() => {
		api
			.model()
			.then(setModelInfo)
			.catch(() => {});
	}, []);

	function fetchTokens(t: string) {
		tokenizeAbortRef.current?.abort();
		clearTimeout(tokenizeDebounceRef.current);

		if (!t) {
			setTokens([]);
			return;
		}

		const controller = new AbortController();
		tokenizeAbortRef.current = controller;

		tokenizeDebounceRef.current = setTimeout(async () => {
			try {
				const res = await api.tokenize(t, controller.signal);
				if (!controller.signal.aborted) setTokens(res.tokens);
			} catch {
				/* aborted or failed */
			}
		}, 80);
	}

	function fetchPrediction(t: string, sp: string, temp: number, tk: number) {
		predictAbortRef.current?.abort();

		if (!t.trim()) {
			setDistribution(null);
			setLoading(false);
			return;
		}

		const controller = new AbortController();
		predictAbortRef.current = controller;
		setLoading(true);
		setError(null);

		api
			.predict(t, sp || null, temp, tk, controller.signal)
			.then((res) => {
				if (!controller.signal.aborted) setDistribution(res.distribution);
			})
			.catch((e) => {
				if (controller.signal.aborted || e instanceof SupersededError) return;
				setError(e instanceof Error ? e.message : "Prediction failed");
			})
			.finally(() => {
				if (!controller.signal.aborted) setLoading(false);
			});
	}

	function schedulePrediction(t: string, sp: string, temp: number, tk: number, delay = 400) {
		clearTimeout(predictDebounceRef.current);
		predictDebounceRef.current = setTimeout(() => fetchPrediction(t, sp, temp, tk), delay);
	}

	function handleTextChange(newText: string, newCharOffset: number) {
		setText(newText);
		setCharOffset(newCharOffset);
		fetchTokens(newText);
		schedulePrediction(newText, systemPrompt, temperature, topK);
	}

	function selectToken(token: TokenCandidate) {
		const s = stateRef.current;
		const newText = s.text + token.text;
		setText(newText);
		setCharOffset(newText.length);
		fetchTokens(newText);
		clearTimeout(predictDebounceRef.current);
		fetchPrediction(newText, s.systemPrompt, s.temperature, s.topK);
		requestAnimationFrame(() => editorRef.current?.focus());
	}

	function handleSystemPromptChange(sp: string) {
		setSystemPrompt(sp);
		if (text.trim()) schedulePrediction(text, sp, temperature, topK);
	}

	function handleTemperatureChange(v: number) {
		setTemperature(v);
		const s = stateRef.current;
		if (s.text.trim()) schedulePrediction(s.text, s.systemPrompt, v, topK, 150);
	}

	function handleTopKChange(v: number) {
		setTopK(v);
		const s = stateRef.current;
		if (s.text.trim()) schedulePrediction(s.text, s.systemPrompt, temperature, v, 150);
	}

	function forcePredict() {
		clearTimeout(predictDebounceRef.current);
		fetchPrediction(text, systemPrompt, temperature, topK);
	}

	const selectTokenRef = useRef(selectToken);
	selectTokenRef.current = selectToken;
	const distRef = useRef(distribution);
	distRef.current = distribution;
	const loadingRef = useRef(loading);
	loadingRef.current = loading;

	useEffect(() => {
		function onKey(e: KeyboardEvent) {
			if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
			if (loadingRef.current) return;
			const t = distRef.current?.tokens;
			if (!t?.length) return;
			if (e.key === "Enter") {
				e.preventDefault();
				selectTokenRef.current(t[0]);
				return;
			}
			const n = Number.parseInt(e.key, 10);
			if (n >= 1 && n <= 9 && t[n - 1]) {
				e.preventDefault();
				selectTokenRef.current(t[n - 1]);
			}
		}
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, []);

	return (
		<div className="h-full flex">
			<main className="flex-1 flex flex-col min-h-0">
				<section className="shrink-0 border-b border-zinc-800/60 p-5">
					<TokenEditor
						ref={editorRef}
						text={text}
						tokens={tokens}
						charOffset={charOffset}
						onTextChange={handleTextChange}
						onCharOffsetChange={setCharOffset}
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
						<TokenList tokens={distribution.tokens} onSelect={selectToken} disabled={loading} />
					) : (
						<div className="h-full flex items-center justify-center text-zinc-600 text-sm">
							{loading ? "Processing…" : "Type something to see predictions"}
						</div>
					)}
				</section>
			</main>

			<aside className="w-80 shrink-0 border-l border-zinc-800/60 overflow-y-auto">
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
	);
}
