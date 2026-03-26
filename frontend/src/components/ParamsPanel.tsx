"use client";

import type { Distribution, ModelInfo } from "@/lib/api";

interface Props {
	temperature: number;
	topK: number;
	onTemperatureChange: (v: number) => void;
	onTopKChange: (v: number) => void;
	systemPrompt: string;
	onSystemPromptChange: (v: string) => void;
	modelInfo: ModelInfo | null;
	distribution: Distribution | null;
}

export default function ParamsPanel({
	temperature,
	topK,
	onTemperatureChange,
	onTopKChange,
	systemPrompt,
	onSystemPromptChange,
	modelInfo,
	distribution,
}: Props) {
	return (
		<div className="p-5 space-y-5 text-sm">
			<section>
				<div className="flex items-center justify-between mb-2.5">
					<h3 className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">System Prompt</h3>
					{systemPrompt ? (
						<span className="text-[9px] text-emerald-500/80 bg-emerald-500/10 px-1.5 py-0.5 rounded">Chat Mode</span>
					) : (
						<span className="text-[9px] text-zinc-600 bg-zinc-800/60 px-1.5 py-0.5 rounded">Completion Mode</span>
					)}
				</div>
				<textarea
					value={systemPrompt}
					onChange={(e) => onSystemPromptChange(e.target.value)}
					placeholder="When set, the chat template wraps your input as a user message."
					rows={3}
					className="w-full bg-zinc-900/70 border border-zinc-800 rounded-lg px-3 py-2 text-xs leading-relaxed focus:outline-none focus:ring-1 focus:ring-violet-500/40 placeholder:text-zinc-600"
					style={{ resize: "vertical", minHeight: "3rem", maxHeight: "12rem" }}
				/>
			</section>

			<section>
				<h3 className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-3">Generation</h3>
				<div className="space-y-4">
					<label className="block">
						<div className="flex justify-between mb-1.5">
							<span className="text-zinc-400 text-xs">Temperature</span>
							<span className="font-mono text-zinc-500 text-xs">{temperature.toFixed(2)}</span>
						</div>
						<input
							type="range"
							min={0}
							max={2}
							step={0.01}
							value={temperature}
							onChange={(e) => onTemperatureChange(Number.parseFloat(e.target.value))}
							className="w-full"
						/>
						<div className="flex justify-between text-[10px] text-zinc-600 mt-1">
							<span>Deterministic</span>
							<span>Creative</span>
						</div>
						<p className="text-[10px] text-zinc-700 mt-1">Higher values spread probability across more tokens.</p>
					</label>

					<label className="block">
						<div className="flex justify-between mb-1.5">
							<span className="text-zinc-400 text-xs">Top-K</span>
							<span className="font-mono text-zinc-500 text-xs">{topK}</span>
						</div>
						<input
							type="number"
							min={1}
							max={2000}
							value={topK}
							onChange={(e) => onTopKChange(Math.max(1, Number.parseInt(e.target.value, 10) || 200))}
							className="w-full bg-zinc-900/70 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-violet-500/40"
						/>
						<p className="text-[10px] text-zinc-700 mt-1">How many candidate tokens to display.</p>
					</label>
				</div>
			</section>

			{distribution && (
				<section>
					<h3 className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-2.5">Inference</h3>
					<dl className="space-y-1.5 text-xs">
						{distribution.prefill_ms != null && (
							<>
								<Row label="Prefill" value={`${distribution.prefill_ms}ms`} />
								<Row label="Speed" value={`${distribution.prefill_tps} tok/s`} />
							</>
						)}
						{distribution.step_ms != null && <Row label="Step" value={`${distribution.step_ms}ms`} />}
						{distribution.cached && <Row label="Cache" value="Reused" />}
						<Row label="Sequence" value={`${distribution.sequence_length} tokens`} />
						<Row label="Vocab" value={distribution.vocab_size.toLocaleString()} />
					</dl>
				</section>
			)}

			{modelInfo && (
				<section>
					<h3 className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-2.5">Model</h3>
					<p className="text-xs text-zinc-400 font-mono break-all leading-relaxed">
						{modelInfo.model_path.split("/").pop()}
					</p>
					<p className="text-[10px] text-zinc-600 mt-1">
						{formatNum(modelInfo.total_parameters)} params · {modelInfo.bits_per_weight} bits/w ·{" "}
						{modelInfo.vocab_size.toLocaleString()} vocab
					</p>
				</section>
			)}
		</div>
	);
}

function Row({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex justify-between gap-2">
			<dt className="text-zinc-500">{label}</dt>
			<dd className="text-zinc-300 font-mono text-right truncate">{value}</dd>
		</div>
	);
}

function formatNum(n: number): string {
	if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
	if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
	if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
	return String(n);
}
