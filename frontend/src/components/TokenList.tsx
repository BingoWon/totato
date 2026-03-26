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
		return tokens.filter((t) => t.text.toLowerCase().includes(q) || String(t.id).includes(q));
	}, [tokens, query]);

	return (
		<div className={`h-full flex flex-col ${disabled ? "opacity-50 pointer-events-none" : ""}`}>
			<div className="shrink-0 px-5 py-3 border-b border-zinc-800/40 flex items-center gap-3">
				<input
					type="text"
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder="Filter tokens…"
					className="flex-1 bg-zinc-900/60 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-violet-500/40 placeholder:text-zinc-600"
				/>
				<span className="text-[10px] text-zinc-600 shrink-0">
					{filtered.length}/{tokens.length}
				</span>
			</div>

			<div className="flex-1 overflow-y-auto">
				{filtered.map((t) => {
					const pct = t.probability * 100;
					const barW = (t.probability / maxProb) * 100;
					const { display, special } = formatToken(t.text);
					const isTop = t.rank <= 3;

					return (
						<button
							type="button"
							key={t.id}
							onClick={() => onSelect(t)}
							className={`w-full flex items-center gap-3 px-5 text-left text-xs transition-colors hover:bg-zinc-800/50 ${
								isTop ? "py-2.5" : "py-1.5"
							}`}
						>
							<span className={`w-7 text-right font-mono shrink-0 ${isTop ? "text-zinc-400" : "text-zinc-600"}`}>
								{t.rank}
							</span>

							<span
								className={`font-mono w-28 truncate shrink-0 ${
									special ? "text-zinc-500 italic" : isTop ? "text-zinc-100" : "text-zinc-300"
								}`}
								title={`ID: ${t.id}`}
							>
								{display}
							</span>

							<div className="flex-1 h-3 bg-zinc-800/40 rounded-sm overflow-hidden">
								<div
									className="h-full rounded-sm transition-all duration-150"
									style={{ width: `${barW}%`, background: barColor(t.rank) }}
								/>
							</div>

							<span className={`w-14 text-right font-mono shrink-0 ${isTop ? "text-zinc-200" : "text-zinc-400"}`}>
								{pct < 0.01 ? "<.01" : pct < 1 ? pct.toFixed(2) : pct.toFixed(1)}%
							</span>
						</button>
					);
				})}
			</div>

			<div className="shrink-0 px-5 py-2 border-t border-zinc-800/30 text-[10px] text-zinc-600 flex justify-between">
				<span>Click to append token</span>
				<span>Blur editor → 1-9 or Enter to quick-select</span>
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
	if (rank <= 10) return "rgba(251,191,36,0.4)";
	if (rank <= 50) return "rgba(251,146,60,0.35)";
	return "rgba(248,113,113,0.3)";
}
