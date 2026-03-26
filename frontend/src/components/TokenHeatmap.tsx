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
	const topAlt = sel?.alternatives[0];
	const isTopPick = sel && sel.rank === 1;

	return (
		<div className="space-y-3">
			<div className="bg-zinc-900/40 border border-zinc-800/40 rounded-xl px-5 py-4">
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
							className={`rounded px-0.5 cursor-pointer select-none transition-all ${
								selected === i ? "ring-2 ring-white/40 scale-105" : "hover:ring-1 hover:ring-white/20"
							}`}
							style={{ backgroundColor: probBg(token.probability) }}
						>
							{renderText(token.text)}
						</span>
					))}
				</div>
				<p className="text-[10px] text-zinc-600 mt-3">Click any token to see what the model expected</p>
			</div>

			{sel && (
				<div className="bg-zinc-900/40 border border-zinc-800/40 rounded-xl px-5 py-4 space-y-3">
					<div>
						<div className="flex items-baseline gap-2 mb-1">
							<span className="font-mono text-base px-1 rounded" style={{ backgroundColor: probBg(sel.probability) }}>
								{formatDisplay(sel.text)}
							</span>
							<span className="text-sm text-zinc-300">{fmtPct(sel.probability)}</span>
						</div>
						<p className="text-xs text-zinc-500">
							{isTopPick
								? "This was exactly what the model expected."
								: `The model expected "${topAlt?.text.trim()}" instead — this was its #${sel.rank} pick.`}
						</p>
					</div>

					<div className="space-y-1">
						<p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-1.5">Model's top predictions</p>
						{sel.alternatives.map((alt, i) => {
							const maxP = sel.alternatives[0]?.probability ?? 1;
							const isActual = alt.id === sel.id;
							return (
								<div key={alt.id} className="flex items-center gap-2 text-xs h-6">
									<span className="text-zinc-600 w-4 text-right font-mono shrink-0">{i + 1}</span>
									<span
										className={`font-mono w-20 truncate shrink-0 ${isActual ? "text-zinc-100 font-medium" : "text-zinc-500"}`}
									>
										{formatDisplay(alt.text)}
									</span>
									{isActual && <span className="text-[9px] text-zinc-600 shrink-0">← actual</span>}
									<div className="flex-1 h-2 bg-zinc-800/60 rounded-full overflow-hidden">
										<div
											className="h-full rounded-full transition-all duration-300"
											style={{
												width: `${Math.max((alt.probability / maxP) * 100, 0.5)}%`,
												backgroundColor: probBg(alt.probability),
											}}
										/>
									</div>
									<span className="font-mono text-zinc-500 w-16 text-right shrink-0">{fmtPct(alt.probability)}</span>
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
	if (pct >= 99.95) return "~100%";
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
