"use client";

import { useState, useEffect } from "react";
import { api, type TokenCandidate, type Distribution, type ModelInfo, type HistoryEntry } from "@/lib/api";
import TokenList from "./TokenList";
import ParamsPanel from "./ParamsPanel";

export default function Explorer() {
  const [prompt, setPrompt] = useState("");
  const [distribution, setDistribution] = useState<Distribution | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(200);
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.model().then(setModelInfo).catch(() => {});
  }, []);

  useEffect(() => {
    if (!distribution || loading) return;
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Enter") {
        e.preventDefault();
        selectToken(distribution!.tokens[0]);
        return;
      }
      const n = parseInt(e.key);
      if (n >= 1 && n <= 9 && distribution!.tokens[n - 1]) {
        e.preventDefault();
        selectToken(distribution!.tokens[n - 1]);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  async function initSession() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.init(prompt, temperature, topK);
      setDistribution(res.distribution);
      setHistory(res.history);
      setActive(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
    } finally {
      setLoading(false);
    }
  }

  async function selectToken(token: TokenCandidate) {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.step(token.token_id, token.probability, token.rank, temperature, topK);
      setDistribution(res.distribution);
      setHistory(res.history);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function resetSession() {
    await api.reset().catch(() => {});
    setDistribution(null);
    setHistory([]);
    setActive(false);
    setError(null);
  }

  return (
    <div className="h-screen flex flex-col">
      <header className="shrink-0 border-b border-zinc-800/60 px-6 py-3.5 flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold tracking-tight">Token Explorer</h1>
          <p className="text-[11px] text-zinc-500 mt-0.5">Interactive Next Token Prediction</p>
        </div>
        {active && (
          <button
            onClick={resetSession}
            className="text-xs text-zinc-500 hover:text-zinc-300 px-3 py-1.5 rounded-md border border-zinc-800 hover:border-zinc-700 transition-colors"
          >
            Reset
          </button>
        )}
      </header>

      <div className="flex flex-1 min-h-0">
        <main className="flex-1 flex flex-col min-h-0">
          {/* Prompt / Generated Text */}
          <section className="shrink-0 border-b border-zinc-800/60 p-5">
            {!active ? (
              <div className="flex gap-3">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt…"
                  rows={3}
                  className="flex-1 bg-zinc-900/70 border border-zinc-800 rounded-lg px-4 py-3 text-sm resize-none
                    focus:outline-none focus:ring-1 focus:ring-violet-500/40 placeholder:text-zinc-600"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) initSession();
                  }}
                />
                <button
                  onClick={initSession}
                  disabled={loading || !prompt.trim()}
                  className="self-end px-5 py-2.5 bg-violet-600 hover:bg-violet-500 disabled:opacity-30
                    disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
                >
                  {loading ? "Loading…" : "Start"}
                </button>
              </div>
            ) : (
              <div className="bg-zinc-900/40 rounded-lg p-4 text-sm font-mono leading-relaxed max-h-52 overflow-y-auto whitespace-pre-wrap break-all">
                <span className="text-zinc-500">{prompt}</span>
                {history.map((h, i) => (
                  <span
                    key={i}
                    className={`rounded-sm transition-colors hover:ring-1 hover:ring-zinc-500 ${rankBg(h.rank)}`}
                    title={`Rank #${h.rank} · ${(h.probability * 100).toFixed(1)}%`}
                  >
                    {h.text}
                  </span>
                ))}
                <span className="inline-block w-1.5 h-4 bg-violet-500/80 animate-pulse align-text-bottom ml-px rounded-sm" />
              </div>
            )}
            {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
          </section>

          {/* Token probability distribution */}
          <section className="flex-1 min-h-0">
            {distribution ? (
              <TokenList tokens={distribution.tokens} onSelect={selectToken} disabled={loading} />
            ) : (
              <div className="h-full flex items-center justify-center text-zinc-600 text-sm">
                {loading ? "Processing prompt…" : "Enter a prompt to begin exploring"}
              </div>
            )}
          </section>
        </main>

        <aside className="w-72 shrink-0 border-l border-zinc-800/60 overflow-y-auto">
          <ParamsPanel
            temperature={temperature}
            topK={topK}
            onTemperatureChange={setTemperature}
            onTopKChange={setTopK}
            modelInfo={modelInfo}
            distribution={distribution}
            historyLength={history.length}
          />
        </aside>
      </div>
    </div>
  );
}

function rankBg(rank: number): string {
  if (rank === 1) return "bg-emerald-500/15 text-emerald-200";
  if (rank <= 5) return "bg-emerald-500/8";
  if (rank <= 20) return "bg-amber-500/10";
  return "bg-red-500/10";
}
