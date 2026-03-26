import { Fragment } from "react";

export function probBg(p: number): string {
	const log = Math.log10(Math.max(p, 1e-6));
	const t = Math.min(1, Math.max(0, (log + 4) / 4));
	const hue = t * 120;
	return `hsla(${hue}, 75%, 45%, 0.35)`;
}

export function fmtPct(p: number): string {
	const pct = p * 100;
	if (pct >= 99.95) return "~100%";
	if (pct < 0.01) return "<0.01%";
	if (pct < 1) return `${pct.toFixed(2)}%`;
	return `${pct.toFixed(1)}%`;
}

export function formatNum(n: number): string {
	if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
	if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
	if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
	return String(n);
}

export function formatTokenDisplay(text: string): { display: string; special: boolean } {
	if (!text) return { display: "∅", special: true };
	if (/^\s+$/.test(text)) {
		return { display: text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→"), special: true };
	}
	return { display: text, special: false };
}

export function renderNewlines(text: string): React.ReactNode {
	if (!text) return null;
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
