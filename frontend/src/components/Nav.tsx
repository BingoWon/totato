"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
	{ href: "/explorer", label: "Token Explorer" },
	{ href: "/scorer", label: "Likelihood Scorer" },
] as const;

export default function Nav() {
	const pathname = usePathname();

	return (
		<nav className="shrink-0 border-b border-zinc-800/60 px-6 h-11 flex items-center gap-6">
			<span className="text-xs font-semibold tracking-tight text-zinc-400 mr-2">Totato</span>
			{links.map(({ href, label }) => (
				<Link
					key={href}
					href={href}
					className={`text-xs transition-colors ${
						pathname === href ? "text-zinc-100 font-medium" : "text-zinc-500 hover:text-zinc-300"
					}`}
				>
					{label}
				</Link>
			))}
		</nav>
	);
}
