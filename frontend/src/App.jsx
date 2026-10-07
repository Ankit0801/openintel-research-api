import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "https://openintel-research-api.onrender.com";

const SOURCES = [
  {
    id: "github",
    name: "GitHub",
    note: "Code and repositories",
    color: "#a78bfa",
    x: 70,
    y: 70,
  },
  {
    id: "arxiv",
    name: "arXiv",
    note: "Research papers",
    color: "#fbbf24",
    x: 330,
    y: 60,
  },
  {
    id: "openalex",
    name: "OpenAlex",
    note: "Academic knowledge",
    color: "#2dd4bf",
    x: 80,
    y: 240,
  },
  {
    id: "nvd",
    name: "NVD",
    note: "Security advisories",
    color: "#fb7185",
    x: 330,
    y: 240,
  },
];
const DEPTHS = [
  [1, "Focused", "Fastest"],
  [2, "Balanced", "Default"],
  [3, "Deep", "More evidence"],
  [4, "Maximum", "Slowest"],
];
const STAGES = [
  "Searching your sources",
  "Reading and cleaning results",
  "Ranking by meaning",
  "Cross-checking claims",
  "Writing the report",
];

const sourceOf = (name = "") =>
  SOURCES.find((s) => name.toLowerCase().includes(s.id)) || SOURCES[0];

function Constellation({ active, loading, counts = {} }) {
  return (
    <svg
      className={`constellation ${loading ? "is-loading" : ""}`}
      viewBox="0 0 400 300"
      role="img"
      aria-label="Map of selected sources feeding the research engine"
    >
      {SOURCES.map((s) => {
        const on = active.includes(s.id);
        return (
          <line
            key={s.id}
            x1="200"
            y1="150"
            x2={s.x}
            y2={s.y}
            className={on ? "link on" : "link"}
            style={{ stroke: on ? s.color : undefined }}
          />
        );
      })}
      {SOURCES.map((s) => {
        const on = active.includes(s.id);
        const n = counts[s.id];
        return (
          <g key={s.id} className={on ? "node on" : "node"}>
            {on && (
              <circle
                cx={s.x}
                cy={s.y}
                r="26"
                fill={s.color}
                className="halo"
              />
            )}
            <circle cx={s.x} cy={s.y} r="9" fill={on ? s.color : "#2a2f66"} />
            <text x={s.x} y={s.y + 28} textAnchor="middle">
              {s.name}
              {n ? ` · ${n}` : ""}
            </text>
          </g>
        );
      })}
      <circle cx="200" cy="150" r="34" className="core-ring" />
      <circle cx="200" cy="150" r="22" className="core" />
      <text x="200" y="157" textAnchor="middle" className="core-text">
        O
      </text>
    </svg>
  );
}

export default function App() {
  const [question, setQuestion] = useState(
    "How can AI assist research workflows?"
  );
  const [sources, setSources] = useState(["github", "arxiv"]);
  const [maxSources, setMaxSources] = useState(2);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [stage, setStage] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!loading) return;
    setStage(0);
    const t = setInterval(
      () => setStage((s) => Math.min(s + 1, STAGES.length - 1)),
      2600
    );
    return () => clearInterval(t);
  }, [loading]);

  const toggleSource = (id) =>
    setSources((cur) =>
      cur.includes(id) ? cur.filter((i) => i !== id) : [...cur, id]
    );

  const runResearch = async () => {
    if (!question.trim()) return setError("Enter a research question first.");
    if (!sources.length)
      return setError("Select at least one source to search.");
    setLoading(true);
    setError("");
    setReport(null);
    try {
      const res = await fetch(`${API_URL}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          preferred_sources: sources,
          max_sources: maxSources,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Research request failed.");
      setReport(data);
      setTimeout(() => window.scrollTo({ top: 0, behavior: "smooth" }), 50);
    } catch (err) {
      setError(
        err.message ||
          "Can't reach the research service. It may be waking up — try again in a minute."
      );
    } finally {
      setLoading(false);
    }
  };

  const copyReport = async () => {
    const text = [
      question,
      "",
      report.answer,
      "",
      "Key findings",
      ...(report.key_findings || []).map((f, i) => `${i + 1}. ${f}`),
      "",
      "Limitations",
      ...(report.limitations || []).map((l) => `- ${l}`),
      "",
      "Sources",
      ...(report.evidence || []).map(
        (e, i) => `[${i + 1}] ${e.title} ${e.url || ""}`
      ),
    ].join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Couldn't copy to the clipboard.");
    }
  };

  const evidence = report?.evidence || [];
  const counts = evidence.reduce((acc, e) => {
    const id = sourceOf(e.source).id;
    acc[id] = (acc[id] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="shell">
      <div className="aurora" aria-hidden="true" />
      <header className="top">
        <button className="logo" onClick={() => setReport(null)} type="button">
          <span className="logo-mark">O</span> OpenIntel
        </button>
        <span className="live">
          <i /> Online
        </span>
      </header>

      <main>
        {!report && (
          <section className="hero">
            <div className="hero-copy">
              <h1>Ask a question. Get an answer you can trace.</h1>
              <p>
                OpenIntel searches code, papers and security data at once,
                checks the results against each other, and writes a report where
                every claim links back to its source.
              </p>
            </div>
            <Constellation active={sources} loading={loading} />
          </section>
        )}

        {!report && (
          <section className="console" aria-busy={loading}>
            <label className="field-label" htmlFor="q">
              Your question
            </label>
            <textarea
              id="q"
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What do you want to find out?"
              disabled={loading}
            />

            <div className="row">
              <div className="group">
                <span className="field-label">Search in</span>
                <div className="chips">
                  {SOURCES.map((s) => {
                    const on = sources.includes(s.id);
                    return (
                      <button
                        key={s.id}
                        type="button"
                        disabled={loading}
                        aria-pressed={on}
                        className={`chip ${on ? "on" : ""}`}
                        style={{ "--c": s.color }}
                        onClick={() => toggleSource(s.id)}
                      >
                        <i /> <b>{s.name}</b> <small>{s.note}</small>
                      </button>
                    );
                  })}
                </div>
              </div>
              <div className="group">
                <span className="field-label">How deep</span>
                <div className="depth" role="radiogroup">
                  {DEPTHS.map(([v, name, hint]) => (
                    <button
                      key={v}
                      type="button"
                      role="radio"
                      aria-checked={maxSources === v}
                      disabled={loading}
                      className={maxSources === v ? "on" : ""}
                      onClick={() => setMaxSources(v)}
                    >
                      <b>{name}</b>
                      <small>{hint}</small>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="actions">
              <button
                className="go"
                type="button"
                onClick={runResearch}
                disabled={loading}
              >
                {loading ? "Researching…" : "Start research"}
              </button>
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
            </div>

            {loading && (
              <ol className="stages" aria-live="polite">
                {STAGES.map((s, i) => (
                  <li
                    key={s}
                    className={i < stage ? "done" : i === stage ? "now" : ""}
                  >
                    <span />
                    {s}
                  </li>
                ))}
              </ol>
            )}
          </section>
        )}

        {report && (
          <section className="report">
            <div className="report-bar">
              <button
                className="ghost"
                type="button"
                onClick={() => setReport(null)}
              >
                ← New question
              </button>
              <button className="ghost" type="button" onClick={copyReport}>
                {copied ? "Copied" : "Copy report"}
              </button>
            </div>
            <h1 className="report-q">{question}</h1>

            <div className="report-grid">
              <article className="brief">
                <p className="answer">{report.answer}</p>

                <h2>Key findings</h2>
                <ol className="findings">
                  {report.key_findings?.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ol>

                {!!report.limitations?.length && (
                  <div className="caveats">
                    <h2>Limitations</h2>
                    <ul>
                      {report.limitations.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </article>

              <aside className="trail">
                <Constellation active={Object.keys(counts)} counts={counts} />
                <h2>{evidence.length} sources</h2>
                <div className="evidence">
                  {evidence.map((e, i) => {
                    const s = sourceOf(e.source);
                    const score = Math.round((e.relevance_score || 0) * 100);
                    return (
                      <article
                        key={i}
                        className="ev"
                        style={{ "--c": s.color }}
                      >
                        <header>
                          <span className="tag">
                            <i />
                            {e.source}
                          </span>
                          <span
                            className="score"
                            title="Relevance to your question"
                          >
                            <em style={{ width: `${score}%` }} />
                            {score}%
                          </span>
                        </header>
                        <h3>
                          [{i + 1}] {e.title}
                        </h3>
                        <p>{e.content}</p>
                        {e.url && (
                          <a href={e.url} target="_blank" rel="noreferrer">
                            Open source ↗
                          </a>
                        )}
                      </article>
                    );
                  })}
                </div>
              </aside>
            </div>
          </section>
        )}
      </main>

      <footer>
        OpenIntel · retrieval, validation and synthesis by multiple agents
      </footer>
    </div>
  );
}
