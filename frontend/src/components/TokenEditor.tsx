"use client";

import { Fragment, forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export interface TokenSpan {
	id: number;
	text: string;
}

interface Props {
	text: string;
	tokens: TokenSpan[];
	cursor: number;
	onTextChange: (text: string, charOffset: number) => void;
	onCursorChange: (tokenIndex: number) => void;
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
	{ text, tokens, cursor, onTextChange, onCursorChange, onForcePredict },
	ref,
) {
	const inputRef = useRef<HTMLInputElement>(null);
	const scrollRef = useRef<HTMLDivElement>(null);
	const [focused, setFocused] = useState(false);

	useImperativeHandle(ref, () => ({ focus: () => inputRef.current?.focus() }));

	useEffect(() => {
		inputRef.current?.focus();
	}, []);

	useEffect(() => {
		const el = scrollRef.current?.querySelector("[data-cursor]");
		el?.scrollIntoView({ block: "nearest", inline: "nearest" });
	});

	function co() {
		return tokenCursorToCharOffset(tokens, cursor);
	}

	function insertAt(chars: string) {
		const offset = co();
		onTextChange(text.slice(0, offset) + chars + text.slice(offset), offset + chars.length);
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
				onCursorChange(Math.max(0, cursor - 1));
				break;
			case "ArrowRight":
				e.preventDefault();
				onCursorChange(Math.min(tokens.length, cursor + 1));
				break;
			case "Home":
				e.preventDefault();
				onCursorChange(0);
				break;
			case "End":
				e.preventDefault();
				onCursorChange(tokens.length);
				break;
			case "Backspace": {
				e.preventDefault();
				if (cursor > 0) {
					const start = tokenCursorToCharOffset(tokens, cursor - 1);
					onTextChange(text.slice(0, start) + text.slice(co()), start);
				}
				break;
			}
			case "Delete": {
				e.preventDefault();
				if (cursor < tokens.length) {
					const end = tokenCursorToCharOffset(tokens, cursor + 1);
					onTextChange(text.slice(0, co()) + text.slice(end), co());
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
		onCursorChange(e.clientX - rect.left < rect.width / 2 ? index : index + 1);
		inputRef.current?.focus();
	}

	return (
		<div
			ref={scrollRef}
			role="textbox"
			tabIndex={-1}
			onClick={() => {
				onCursorChange(tokens.length);
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
				{tokens.length === 0 ? (
					<span className="text-zinc-600 pointer-events-none select-none">Start typing to explore tokens…</span>
				) : (
					<>
						{focused && cursor === 0 && <CursorLine />}
						{tokens.map((token, i) => (
							<Fragment key={`t${token.id}p${tokenCursorToCharOffset(tokens, i)}`}>
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
									{renderTokenText(token.text)}
								</span>
								{focused && cursor === i + 1 && <CursorLine />}
							</Fragment>
						))}
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

function renderTokenText(text: string): React.ReactNode {
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

export function tokenCursorToCharOffset(tokens: TokenSpan[], cursor: number): number {
	let offset = 0;
	for (let i = 0; i < cursor && i < tokens.length; i++) {
		offset += tokens[i].text.length;
	}
	return offset;
}

export function charOffsetToTokenCursor(tokens: TokenSpan[], offset: number): number {
	if (offset <= 0) return 0;
	let acc = 0;
	for (let i = 0; i < tokens.length; i++) {
		acc += tokens[i].text.length;
		if (acc >= offset) return i + 1;
	}
	return tokens.length;
}
