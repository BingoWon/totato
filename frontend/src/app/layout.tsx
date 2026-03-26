import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
	title: "Totato",
	description: "Token Explorer & Likelihood Scorer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en">
			<body className="antialiased bg-zinc-950 text-zinc-100 h-screen flex flex-col">
				<Nav />
				<div className="flex-1 min-h-0">{children}</div>
			</body>
		</html>
	);
}
