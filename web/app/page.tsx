"use client";
import { FormEvent, useState } from "react";

type Citation = { chunk_id: string; quote: string; title: string; url: string; page?: number; section?: string };
type Answer = { status: "answered" | "abstained"; answer: string; citations: Citation[]; trace_id: string; corpus_release_id: string; retrieved: { chunk_id: string; hybrid_score: number; rerank_score: number }[] };

/** Renders a transparent, source-first client for the FastAPI governed-answer endpoint. */
export default function Home() {
  // The UI retains only the latest governed response; source evidence comes from the API, never the browser.
  const [question, setQuestion] = useState("What is the purpose of the NIST AI RMF?");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  async function submit(event: FormEvent) {
    // Submit one question to FastAPI and wait for its retrieval, citation, and abstention gates to finish.
    event.preventDefault(); setLoading(true); setAnswer(null);
    const endpoint = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const response = await fetch(`${endpoint}/v1/query`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    setAnswer(await response.json()); setLoading(false);
  }
  return <main><header><p className="eyebrow">EVIDENCE-GROUNDED RAG</p><h1>AI Governance Copilot</h1><p>Every factual answer must be supported by approved source evidence.</p></header>
    <form onSubmit={submit}><label htmlFor="question">Ask about the approved AI-governance corpus</label><textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} /><button disabled={loading}>{loading ? "Checking evidence..." : "Ask with citations"}</button></form>
    {/* This block deliberately distinguishes a verified answer from a safe abstention. */}
    {answer && <section className={answer.status}><p className="badge">{answer.status === "answered" ? "Citation-verified response" : "Insufficient evidence"}</p><p className="answer">{answer.answer}</p><p className="meta">Corpus {answer.corpus_release_id} · Trace {answer.trace_id}</p>
      <h2>Source evidence</h2>{answer.citations.length ? answer.citations.map((c) => <article key={`${c.chunk_id}-${c.quote}`}><a href={c.url}>{c.title}</a><blockquote>{c.quote}</blockquote><small>{c.section || "Source excerpt"}{c.page ? ` · page ${c.page}` : ""}</small></article>) : <p>No citation is shown because the system did not release an unsupported answer.</p>}
      <details><summary>Retrieval diagnostics</summary><pre>{JSON.stringify(answer.retrieved, null, 2)}</pre></details></section>}</main>;
}
