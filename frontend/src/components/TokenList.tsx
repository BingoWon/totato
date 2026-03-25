"use client";

import { useMemo, useState } from "react";
import type { TokenCandidate } from "@/lib/api";

interface Props {
	tokens: TokenCandidate[];
	onSelect: (token: TokenCandidate) => void;
	disabled: boolean;
}

export default function TokenList({ tokens, onSelect, disabled }: Props) {
	const [query, setQuery] = useState("");

	const maxProb = tokens[0]?.probability ?? 1;

	const filtered = useMemo(() => {
		if (!query) return tokens;
		const q = query.toLowerCase();
		return tokens.filter((t) => t.text.toLowerCase().includes(q) || String(t.token_id).includes(q));
	}, [tokens, query]);

	return (
		<div className={`h-full flex flex-col ${disabled ? "opacity-60 pointer-events-none" : ""}`}>
			<div className="shrink-0 px-4 py-3 border-b border-zinc-800/40">
				<input
					type="text"
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder="Search tokens…"
					className="w-full bg-zinc-900/60 border border-zinc-800 rounded-md px-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-500/40 placeholder:text-zinc-600"
				/>
			</div>

			<div className="shrink-0 grid grid-cols-[3rem_minmax(5rem,1fr)_1fr_4.5rem_4rem] gap-2 px-4 py-2 text-[10px] text-zinc-500 uppercase tracking-wider border-b border-zinc-800/30">
				<span>Rank</span>
				<span>Token</span>
				<span>Probability</span>
				<span className="text-right">%</span>
				<span className="text-right">Logit</span>
			</div>

			<div className="flex-1 overflow-y-auto">
				{filtered.map((t) => {
					const pct = t.probability * 100;
					const barW = (t.probability / maxProb) * 100;
					const { display, special } = formatToken(t.text);

					return (
						<button
							type="button"
							key={t.token_id}
							onClick={() => onSelect(t)}
							className="w-full grid grid-cols-[3rem_minmax(5rem,1fr)_1fr_4.5rem_4rem] gap-2 items-center px-4 py-1.5 text-left text-xs hover:bg-zinc-800/50 transition-colors group"
						>
							<span className="text-zinc-600 font-mono">{t.rank}</span>

							<span
								className={`font-mono truncate ${special ? "text-zinc-500 italic" : "text-zinc-200"}`}
								title={`ID: ${t.token_id}`}
							>
								{display}
							</span>

							<div className="h-4 bg-zinc-800/60 rounded-sm overflow-hidden">
								<div
									className="h-full rounded-sm transition-all duration-150"
									style={{ width: `${barW}%`, background: barColor(t.rank) }}
								/>
							</div>

							<span className="text-right font-mono text-zinc-300">
								{pct < 0.01 ? "<.01" : pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}
							</span>

							<span className="text-right font-mono text-zinc-600">{t.logit.toFixed(1)}</span>
						</button>
					);
				})}
			</div>

			<div className="shrink-0 px-4 py-2 border-t border-zinc-800/30 text-[10px] text-zinc-600 flex justify-between">
				<span>
					{filtered.length} / {tokens.length} tokens shown
				</span>
				<span>Click to select · Blur editor for 1-9 / Enter</span>
			</div>
		</div>
	);
}

function formatToken(text: string): { display: string; special: boolean } {
	if (!text) return { display: "∅", special: true };
	if (/^\s+$/.test(text)) {
		const d = text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
		return { display: d, special: true };
	}
	return { display: text, special: false };
}

function barColor(rank: number): string {
	if (rank === 1) return "rgba(52,211,153,0.7)";
	if (rank <= 3) return "rgba(52,211,153,0.5)";
	if (rank <= 10) return "rgba(251,191,36,0.45)";
	if (rank <= 50) return "rgba(251,146,60,0.4)";
	return "rgba(248,113,113,0.35)";
}
