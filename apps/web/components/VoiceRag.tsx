"use client";

/**
 * The full voice loop: hold-to-talk mic → /transcribe (Sarvam) → editable
 * transcript → /ask → answer + citations + per-stage latency waterfall.
 *
 * Typing a question is a first-class path, not a fallback — the measured
 * budget starts at query text either way, and it keeps the demo alive even
 * with no STT key configured.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_GOARAG_API || "http://127.0.0.1:8099";

type Timing = { stage: string; ms: number };
type Citation = { n: number; passage_id: string; text: string };
type AskResponse = {
  trace_id: string;
  answer: { text: string; path: string; citations: Citation[]; lang: string };
  retrieved: { passage_id: string; text: string; strategy: string; lang: string }[];
  timings: Timing[];
  budget_ms: number;
  total_ms: number;
};

/**
 * Real queries from the indexed MSMARCO-XI subset, verified to retrieve their
 * gold passage — plus one deliberate off-topic query to show the guardrail
 * refusing on camera. Invented questions ("capital of India") are NOT in this
 * corpus and would demo a refusal or a wrong hit.
 */
const SAMPLES = [
  "what is a corporation?",
  "honesty or integrity definition",
  "कॉर्पोरेशन क्या है?",
  "கட்டுமானக் கடன்கள் எவ்வாறு செயல்படுகின்றன?",
  "what is the weather in goa tomorrow",
];

const STAGE_COLORS: Record<string, string> = {
  guard_input: "#FBF7EC",
  lang_detect: "#FBF7EC",
  embed: "#FFD400",
  guard_domain: "#FBF7EC",
  retrieve: "#EC1E79",
  extract: "#7ED957",
};

export default function VoiceRag() {
  const [query, setQuery] = useState("");
  const [recording, setRecording] = useState(false);
  const [sttInfo, setSttInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{ q: string; ms: number; path: string }[]>([]);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  /* ------------------------------------------------------------------ ask */
  const ask = useCallback(async (q: string) => {
    const text = q.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    try {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      const d: AskResponse = await r.json();
      setResp(d);
      setHistory((h) => [{ q: text, ms: d.budget_ms, path: d.answer.path }, ...h].slice(0, 8));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  /* ------------------------------------------------------------------ mic */
  const startRec = useCallback(async () => {
    setError(null);
    setSttInfo(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      mr.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (blob.size < 1000) return; // accidental tap
        setBusy(true);
        try {
          const form = new FormData();
          form.append("file", blob, "query.webm");
          const r = await fetch(`${API}/transcribe`, { method: "POST", body: form });
          const d = await r.json();
          if (!r.ok) throw new Error(d.detail || `STT ${r.status}`);
          setQuery(d.text);
          setSttInfo(`${d.provider} · ${d.lang} · ${(d.confidence * 100) | 0}%`);
          if (d.text) await ask(d.text);
        } catch (e) {
          setBusy(false);
          setError(
            (e instanceof Error ? e.message : "Transcription failed.") +
              " — type your question instead."
          );
        }
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
    } catch {
      setError("Microphone access denied — type your question instead.");
    }
  }, [ask]);

  const stopRec = useCallback(() => {
    mediaRef.current?.stop();
    mediaRef.current = null;
    setRecording(false);
  }, []);

  useEffect(() => () => mediaRef.current?.stop(), []);

  const budget = resp?.budget_ms ?? 0;
  const under = budget > 0 && budget < 200;

  /* ------------------------------------------------------------------ ui */
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
      <section className="space-y-4">
        {/* input row */}
        <div className="flex gap-2">
          <button
            className={`btn ${recording ? "bg-pink text-cream shadow-[4px_4px_0_0_#111]" : "btn-primary"} shrink-0`}
            onPointerDown={startRec}
            onPointerUp={stopRec}
            onPointerLeave={() => recording && stopRec()}
            disabled={busy}
            title="Hold to talk"
          >
            {recording ? "● listening…" : "🎤 hold to talk"}
          </button>
          <input
            className="field"
            placeholder="…or type: English / हिन्दी / தமிழ்"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask(query)}
            disabled={busy}
          />
          <button className="btn-pink shrink-0" onClick={() => ask(query)} disabled={busy || !query.trim()}>
            {busy ? "…" : "ask"}
          </button>
        </div>

        {sttInfo && (
          <p className="font-mono text-[11px] uppercase tracking-wider text-cream/60">
            transcribed via {sttInfo}
          </p>
        )}
        {error && (
          <p className="rounded-xl border-2 border-pink/50 bg-pink/10 p-3 font-mono text-xs text-cream/90">
            {error}
          </p>
        )}

        {/* sample questions */}
        <div className="flex flex-wrap gap-2">
          {SAMPLES.map((s) => (
            <button
              key={s}
              className="rounded-full border border-cream/25 bg-white/5 px-3 py-1.5 font-mono text-[11px] text-cream/75 transition hover:border-yellow hover:text-cream"
              onClick={() => {
                setQuery(s);
                ask(s);
              }}
              disabled={busy}
            >
              {s}
            </button>
          ))}
        </div>

        {/* answer */}
        {resp && (
          <div className="space-y-4 rounded-2xl border-2 border-ink bg-white/5 p-5 shadow-[6px_6px_0_0_rgba(17,17,17,0.55)]">
            <div className="flex flex-wrap items-center gap-3">
              <span
                className={`rounded-full px-3 py-1 font-mono text-[11px] font-bold uppercase tracking-wider ${
                  resp.answer.path === "refusal"
                    ? "bg-pink text-cream"
                    : "bg-yellow text-ink"
                }`}
              >
                {resp.answer.path}
              </span>
              <span
                className={`font-mono text-sm font-bold ${under ? "text-[#7ED957]" : "text-yellow"}`}
              >
                {budget.toFixed(1)} ms
              </span>
              <span className="font-mono text-[11px] text-cream/50">
                budget: query → answer · target &lt;200 ms
              </span>
            </div>

            <p className="font-display text-2xl leading-snug text-cream">{resp.answer.text}</p>

            {resp.answer.citations.length > 0 && (
              <div className="space-y-2">
                {resp.answer.citations.map((c) => (
                  <details key={c.n} className="rounded-xl border border-cream/20 bg-black/20 p-3">
                    <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-wider text-yellow">
                      [{c.n}] passage {c.passage_id}
                    </summary>
                    <p className="mt-2 font-mono text-xs leading-relaxed text-cream/80">{c.text}</p>
                  </details>
                ))}
              </div>
            )}

            {/* latency waterfall */}
            <div className="space-y-1.5">
              {resp.timings.map((t) => {
                const w = Math.max(1.5, (t.ms / Math.max(budget, 1)) * 100);
                return (
                  <div key={t.stage} className="flex items-center gap-2">
                    <span className="w-32 shrink-0 truncate text-right font-mono text-[10px] text-cream/60">
                      {t.stage}
                    </span>
                    <div className="h-3 flex-1 overflow-hidden rounded-sm bg-black/25">
                      <div
                        className="h-full rounded-sm"
                        style={{
                          width: `${Math.min(w, 100)}%`,
                          background: STAGE_COLORS[t.stage] || "#FFD400",
                        }}
                      />
                    </div>
                    <span className="w-16 shrink-0 font-mono text-[10px] text-cream/70">
                      {t.ms.toFixed(2)} ms
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* history rail */}
      <aside className="space-y-3">
        <h2 className="font-mono text-[11px] font-bold uppercase tracking-[0.2em] text-cream/60">
          session history
        </h2>
        {history.length === 0 && (
          <p className="font-mono text-xs text-cream/40">
            Ask something — every query lands here with its latency.
          </p>
        )}
        {history.map((h, i) => (
          <button
            key={i}
            className="block w-full rounded-xl border border-cream/15 bg-white/5 p-3 text-left transition hover:border-yellow/60"
            onClick={() => {
              setQuery(h.q);
              ask(h.q);
            }}
          >
            <p className="truncate font-mono text-xs text-cream/85">{h.q}</p>
            <p className="mt-1 font-mono text-[10px] uppercase tracking-wider text-cream/50">
              {h.path} · <span className={h.ms < 200 ? "text-[#7ED957]" : "text-yellow"}>{h.ms.toFixed(1)} ms</span>
            </p>
          </button>
        ))}
      </aside>
    </div>
  );
}
