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
		<div className="p-5 space-y-6 text-sm">
			<Section title="System Prompt">
				<textarea
					value={systemPrompt}
					onChange={(e) => onSystemPromptChange(e.target.value)}
					placeholder="Optional. When set, chat template is applied automatically."
					rows={3}
					className="w-full bg-zinc-900/70 border border-zinc-800 rounded-md px-3 py-2 text-xs leading-relaxed focus:outline-none focus:ring-1 focus:ring-violet-500/40 placeholder:text-zinc-600"
					style={{ resize: "vertical", minHeight: "3rem", maxHeight: "12rem" }}
				/>
				{modelInfo && (
					<p className="text-[10px] text-zinc-600 mt-1.5">
						{modelInfo.has_chat_template ? "Chat template available" : "No chat template detected"}
					</p>
				)}
			</Section>

			<Section title="Parameters">
				<div className="space-y-4">
					<label className="block">
						<div className="flex justify-between mb-2">
							<span className="text-zinc-400">Temperature</span>
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
							<span>Greedy</span>
							<span>Creative</span>
							<span>Wild</span>
						</div>
					</label>

					<label className="block">
						<span className="text-zinc-400 block mb-2">Top-K</span>
						<input
							type="number"
							min={1}
							max={2000}
							value={topK}
							onChange={(e) => onTopKChange(Math.max(1, Number.parseInt(e.target.value, 10) || 200))}
							className="w-full bg-zinc-900/70 border border-zinc-800 rounded-md px-3 py-1.5 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-violet-500/40"
						/>
					</label>
				</div>
			</Section>

			{distribution && (
				<Section title="Inference">
					<dl className="space-y-2">
						{distribution.prefill_ms != null && (
							<>
								<Stat label="Prefill" value={`${distribution.prefill_ms} ms`} />
								<Stat label="Prefill Speed" value={`${distribution.prefill_tps} tok/s`} />
							</>
						)}
						{distribution.step_ms != null && <Stat label="Step Latency" value={`${distribution.step_ms} ms`} />}
						{distribution.cached && <Stat label="Cache" value="Logits reused" />}
						<Stat label="Sequence Length" value={`${distribution.sequence_length} tokens`} />
						<Stat label="Vocab Size" value={distribution.vocab_size.toLocaleString()} />
					</dl>
				</Section>
			)}

			{modelInfo && (
				<Section title="Model">
					<dl className="space-y-2">
						<Stat label="Path" value={modelInfo.model_path.split("/").pop() ?? modelInfo.model_path} />
						<Stat label="Parameters" value={formatNum(modelInfo.total_parameters)} />
						<Stat label="Bits/Weight" value={String(modelInfo.bits_per_weight)} />
						<Stat label="Vocab Size" value={modelInfo.vocab_size.toLocaleString()} />
					</dl>
				</Section>
			)}
		</div>
	);
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
	return (
		<section>
			<h3 className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-3">{title}</h3>
			{children}
		</section>
	);
}

function Stat({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex justify-between gap-2">
			<dt className="text-zinc-500 shrink-0">{label}</dt>
			<dd className="text-zinc-300 font-mono text-xs text-right truncate">{value}</dd>
		</div>
	);
}

function formatNum(n: number): string {
	if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
	if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
	if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
	return String(n);
}
