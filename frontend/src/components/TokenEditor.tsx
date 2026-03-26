"use client";

import { Fragment, forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { TokenSpan } from "@/lib/api";
import { renderNewlines } from "@/lib/format";

interface Props {
	text: string;
	tokens: TokenSpan[];
	charOffset: number;
	onTextChange: (text: string, charOffset: number) => void;
	onCharOffsetChange: (charOffset: number) => void;
	onForcePredict: () => void;
}

export interface TokenEditorHandle {
	focus: () => void;
}

const TOKEN_COLORS = [
	"bg-sky-500/20 hover:bg-sky-500/30",
	"bg-emerald-500/20 hover:bg-emerald-500/30",
	"bg-amber-500/20 hover:bg-amber-500/30",
	"bg-pink-500/20 hover:bg-pink-500/30",
	"bg-violet-500/20 hover:bg-violet-500/30",
];

const TokenEditor = forwardRef<TokenEditorHandle, Props>(function TokenEditor(
	{ text, tokens, charOffset, onTextChange, onCharOffsetChange, onForcePredict },
	ref,
) {
	const inputRef = useRef<HTMLInputElement>(null);
	const scrollRef = useRef<HTMLDivElement>(null);
	const [focused, setFocused] = useState(false);

	useImperativeHandle(ref, () => ({ focus: () => inputRef.current?.focus() }));

	useEffect(() => {
		inputRef.current?.focus();
	}, []);

	const offsets = useMemo(() => {
		const arr = [0];
		for (const t of tokens) arr.push(arr[arr.length - 1] + t.text.length);
		return arr;
	}, [tokens]);

	const synced = useMemo(() => {
		if (!tokens.length) return false;
		if (offsets[tokens.length] !== text.length) return false;
		return tokens.map((t) => t.text).join("") === text;
	}, [tokens, text, offsets]);

	const tokenCursor = synced ? charOffsetToTokenCursor(tokens, offsets, charOffset) : -1;

	// biome-ignore lint/correctness/useExhaustiveDependencies: charOffset/focused drive cursor DOM changes
	useEffect(() => {
		const el = scrollRef.current?.querySelector("[data-cursor]");
		el?.scrollIntoView({ block: "nearest", inline: "nearest" });
	}, [charOffset, focused]);

	function insertAt(chars: string) {
		onTextChange(text.slice(0, charOffset) + chars + text.slice(charOffset), charOffset + chars.length);
	}

	function handleKeyDown(e: React.KeyboardEvent) {
		if (e.nativeEvent.isComposing) return;

		if (e.metaKey || e.ctrlKey) {
			switch (e.key) {
				case "Backspace":
					e.preventDefault();
					onTextChange("", 0);
					return;
				case "c":
					navigator.clipboard.writeText(text);
					return;
				case "Enter":
					e.preventDefault();
					onForcePredict();
					return;
			}
			return;
		}

		switch (e.key) {
			case "ArrowLeft":
				e.preventDefault();
				if (synced && tokenCursor > 0) {
					onCharOffsetChange(offsets[tokenCursor - 1]);
				} else {
					onCharOffsetChange(Math.max(0, charOffset - 1));
				}
				break;
			case "ArrowRight":
				e.preventDefault();
				if (synced && tokenCursor < tokens.length) {
					onCharOffsetChange(offsets[tokenCursor + 1]);
				} else {
					onCharOffsetChange(Math.min(text.length, charOffset + 1));
				}
				break;
			case "Home":
				e.preventDefault();
				onCharOffsetChange(0);
				break;
			case "End":
				e.preventDefault();
				onCharOffsetChange(text.length);
				break;
			case "Backspace": {
				e.preventDefault();
				if (synced && tokenCursor > 0) {
					const start = offsets[tokenCursor - 1];
					onTextChange(text.slice(0, start) + text.slice(charOffset), start);
				} else if (charOffset > 0) {
					onTextChange(text.slice(0, charOffset - 1) + text.slice(charOffset), charOffset - 1);
				}
				break;
			}
			case "Delete": {
				e.preventDefault();
				if (synced && tokenCursor < tokens.length) {
					const end = offsets[tokenCursor + 1];
					onTextChange(text.slice(0, charOffset) + text.slice(end), charOffset);
				} else if (charOffset < text.length) {
					onTextChange(text.slice(0, charOffset) + text.slice(charOffset + 1), charOffset);
				}
				break;
			}
			case "Enter":
				e.preventDefault();
				insertAt("\n");
				break;
			case "Tab":
				e.preventDefault();
				insertAt("\t");
				break;
			case "Escape":
				e.preventDefault();
				inputRef.current?.blur();
				break;
		}
	}

	function handleInput(e: React.FormEvent<HTMLInputElement>) {
		if ((e.nativeEvent as InputEvent).isComposing) return;
		const el = e.target as HTMLInputElement;
		const val = el.value;
		el.value = "";
		if (val) insertAt(val);
	}

	function handleCompositionEnd() {
		requestAnimationFrame(() => {
			const el = inputRef.current;
			if (el?.value) {
				insertAt(el.value);
				el.value = "";
			}
		});
	}

	function handlePaste(e: React.ClipboardEvent) {
		e.preventDefault();
		const pasted = e.clipboardData.getData("text");
		if (pasted) insertAt(pasted);
	}

	function handleTokenClick(e: React.MouseEvent, index: number) {
		e.stopPropagation();
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const target = e.clientX - rect.left < rect.width / 2 ? index : index + 1;
		onCharOffsetChange(offsets[target]);
		inputRef.current?.focus();
	}

	return (
		<div
			ref={scrollRef}
			role="textbox"
			tabIndex={-1}
			onClick={() => {
				onCharOffsetChange(text.length);
				inputRef.current?.focus();
			}}
			onKeyDown={() => inputRef.current?.focus()}
			className={`relative min-h-24 max-h-80 overflow-y-auto rounded-lg border cursor-text transition-colors ${
				focused
					? "border-violet-500/40 ring-1 ring-violet-500/15 bg-zinc-900/80"
					: "border-zinc-800 bg-zinc-900/60 hover:border-zinc-700"
			}`}
		>
			<input
				ref={inputRef}
				onInput={handleInput}
				onKeyDown={handleKeyDown}
				onPaste={handlePaste}
				onCompositionEnd={handleCompositionEnd}
				onFocus={() => setFocused(true)}
				onBlur={() => setFocused(false)}
				aria-label="Token editor input"
				autoComplete="off"
				autoCorrect="off"
				spellCheck={false}
				className="absolute w-0 h-0 opacity-0 overflow-hidden"
			/>

			<div className="px-4 py-3 font-mono text-sm leading-relaxed whitespace-pre-wrap break-all min-h-[inherit]">
				{!text ? (
					<span className="text-zinc-600 pointer-events-none select-none">Start typing to explore tokens…</span>
				) : synced ? (
					<>
						{focused && tokenCursor === 0 && <CursorLine />}
						{tokens.map((token, i) => (
							<Fragment key={`${offsets[i]}:${token.id}`}>
								<span
									role="button"
									tabIndex={-1}
									className={`${TOKEN_COLORS[i % TOKEN_COLORS.length]} rounded-[3px] px-px transition-colors cursor-pointer select-none`}
									onClick={(e) => handleTokenClick(e, i)}
									onKeyDown={(e) => {
										if (e.key === "Enter" || e.key === " ") handleTokenClick(e as unknown as React.MouseEvent, i);
									}}
									title={`#${token.id}`}
								>
									{renderNewlines(token.text)}
								</span>
								{focused && tokenCursor === i + 1 && <CursorLine />}
							</Fragment>
						))}
					</>
				) : (
					<>
						{renderNewlines(text.slice(0, charOffset))}
						{focused && <CursorLine />}
						{renderNewlines(text.slice(charOffset))}
					</>
				)}
			</div>
		</div>
	);
});

export default TokenEditor;

function CursorLine() {
	return (
		<span
			data-cursor=""
			className="inline-block w-[2px] h-[1.15em] bg-violet-400 animate-pulse rounded-full align-text-bottom"
		/>
	);
}

function charOffsetToTokenCursor(tokens: TokenSpan[], offsets: number[], offset: number): number {
	if (offset <= 0) return 0;
	for (let i = 0; i < tokens.length; i++) {
		if (offsets[i + 1] >= offset) return i + 1;
	}
	return tokens.length;
}
