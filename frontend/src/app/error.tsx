"use client";

export default function ErrorBoundary({ error, reset }: { error: Error; reset: () => void }) {
	return (
		<div className="h-full flex items-center justify-center">
			<div className="text-center space-y-4 max-w-md px-6">
				<h2 className="text-lg font-semibold text-red-400">Something went wrong</h2>
				<p className="text-sm text-zinc-400">{error.message}</p>
				<button
					type="button"
					onClick={reset}
					className="px-4 py-2 text-sm bg-zinc-800 hover:bg-zinc-700 rounded-md transition-colors"
				>
					Try again
				</button>
			</div>
		</div>
	);
}
