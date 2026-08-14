import type { Metadata } from "next";
import { DM_Serif_Display, Space_Mono } from "next/font/google";
import "./globals.css";

const display = DM_Serif_Display({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display",
});
const mono = Space_Mono({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "GoaRAG · Voice RAG for HACKER HOUSE गोवा 2026",
  description:
    "Speak a question in English, Hindi or Tamil — grounded answers from MS MARCO-XI in under 200 ms. #RAGInGoa",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${mono.variable}`}>
      <body className="min-h-screen text-cream antialiased">{children}</body>
    </html>
  );
}
