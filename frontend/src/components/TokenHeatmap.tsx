"use client";

import { Fragment } from "react";
import type { TokenScore } from "@/lib/api";

interface Props {
	tokens: TokenScore[];
	selected: number | null;
	onSelect: (index: number | null) => void;
}

export default function TokenHeatmap({ tokens, selected, onSelect }: Props) {
	const sel = selected !== null ? tokens[selected] : null;

	return (
		<div className="flex flex-col h-full">
			<div className="flex-1 overflow-y-auto px-5 py-4">
				<div className="font-mono text-sm leading-loose whitespace-pre-wrap break-all">
					{tokens.map((token, i) => (
						<span
							key={i}
							role="button"
							tabIndex={-1}
							onClick={() => onSelect(selected === i ? null : i)}
							onKeyDown={(e) => {
								if (e.key === "Enter" || e.key === " ") onSelect(selected === i ? null : i);
							}}
							className={`rounded-[3px] px-px cursor-pointer select-none transition-all ${
								selected === i ? "ring-2 ring-white/50 scale-110" : "hover:brightness-125"
							}`}
							style={{ backgroundColor: probBg(token.probability) }}
							title={`"${token.text}" — ${fmtPct(token.probability)} (rank ${token.rank})`}
						>
							{renderText(token.text)}
						</span>
					))}
				</div>

				<div className="flex items-center gap-2 mt-4 text-[10px] text-zinc-500">
					<span>Probability:</span>
					<div className="flex h-2.5 rounded-full overflow-hidden w-40">
						{[0, 1, 2, 3, 4].map((i) => (
							<div key={i} className="flex-1" style={{ backgroundColor: probBg(10 ** -(4 - i)) }} />
						))}
					</div>
					<span>Low → High</span>
				</div>
			</div>

			{sel && (
				<div className="shrink-0 border-t border-zinc-800/50 px-5 py-4">
					<div className="flex items-baseline gap-3 mb-3">
						<span className="font-mono text-sm" style={{ backgroundColor: probBg(sel.probability) }}>
							{formatDisplay(sel.text)}
						</span>
						<span className="text-xs text-zinc-400">
							rank {sel.rank} · {fmtPct(sel.probability)} · log {sel.log_prob.toFixed(2)}
						</span>
					</div>
					<div className="space-y-1.5">
						{sel.alternatives.map((alt, i) => {
							const maxP = sel.alternatives[0]?.probability ?? 1;
							return (
								<div key={alt.id} className="flex items-center gap-2 text-xs">
									<span className="text-zinc-600 w-4 text-right font-mono">{i + 1}</span>
									<span className={`font-mono w-24 truncate ${alt.id === sel.id ? "text-zinc-100" : "text-zinc-400"}`}>
										{formatDisplay(alt.text)}
									</span>
									<div className="flex-1 h-3 bg-zinc-800/60 rounded-sm overflow-hidden">
										<div
											className="h-full rounded-sm"
											style={{
												width: `${(alt.probability / maxP) * 100}%`,
												backgroundColor: probBg(alt.probability),
											}}
										/>
									</div>
									<span className="font-mono text-zinc-500 w-16 text-right">{fmtPct(alt.probability)}</span>
								</div>
							);
						})}
					</div>
				</div>
			)}
		</div>
	);
}

function probBg(p: number): string {
	const log = Math.log10(Math.max(p, 1e-6));
	const t = Math.min(1, Math.max(0, (log + 4) / 4));
	const hue = t * 120;
	return `hsla(${hue}, 75%, 45%, 0.35)`;
}

function fmtPct(p: number): string {
	const pct = p * 100;
	if (pct < 0.01) return "<0.01%";
	if (pct < 1) return `${pct.toFixed(2)}%`;
	return `${pct.toFixed(1)}%`;
}

function formatDisplay(text: string): string {
	if (!text) return "∅";
	if (/^\s+$/.test(text)) return text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
	return text;
}

function renderText(text: string): React.ReactNode {
	if (!text) return <span className="text-zinc-600 text-[10px]">∅</span>;
	if (!text.includes("\n")) return text;
	const parts = text.split("\n");
	return parts.map((part, idx) => (
		<Fragment key={`ln${idx.toString()}`}>
			{part}
			{idx < parts.length - 1 && (
				<>
					<span className="text-zinc-600/50 text-[10px]">↵</span>
					{"\n"}
				</>
			)}
		</Fragment>
	));
}
