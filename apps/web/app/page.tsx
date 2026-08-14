import Image from "next/image";
import VoiceRag from "@/components/VoiceRag";

export default function Home() {
  return (
    <>
      {/* Same hero treatment as Task 1 — cross-submission identity. */}
      <div className="relative h-28 w-full overflow-hidden border-b-4 border-ink sm:h-40">
        {/* beach-props, not palms-banner: the palms artwork has the wordmark
            baked into its centre, which a crop slices in half and duplicates
            the real lockup directly below. */}
        <Image
          src="/scenes/beach-props.png"
          alt=""
          fill
          priority
          sizes="100vw"
          className="scale-[1.04] object-cover object-center"
        />
      </div>

      <main className="mx-auto w-full max-w-5xl px-4 py-6 sm:px-6 sm:py-10">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="sr-only">GoaRAG — voice RAG for HACKER HOUSE गोवा 2026</h1>
            <span className="relative inline-block h-10 sm:h-14">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/brand/wordmark.png" alt="Hacker House" className="h-full w-auto" />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/brand/goa-sticker.png"
                alt="गोवा"
                className="absolute left-[53%] top-1/2 h-[88%] w-auto -translate-x-1/2 -translate-y-1/2 -rotate-3"
              />
            </span>
            <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.2em] text-cream/70">
              GoaRAG · Voice-enabled RAG · #RAGInGoa
            </p>
          </div>
          <p className="max-w-sm font-mono text-[11px] leading-relaxed text-cream/60">
            Speak a question in English, हिन्दी or தமிழ். Grounded answers straight
            from MS MARCO-XI passages — retrieval to answer in well under 200 ms,
            with the stage-by-stage waterfall to prove it.
          </p>
        </header>

        <VoiceRag />

        <footer className="mt-10 border-t border-cream/15 pt-4 font-mono text-[11px] text-cream/45">
          HH Goa 2026 · Task 2 · extractive-first RAG · everything hot runs in one
          process, in RAM — the fastest network call is none.
        </footer>
      </main>
    </>
  );
}
