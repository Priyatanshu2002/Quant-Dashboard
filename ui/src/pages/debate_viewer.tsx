import { useEffect, useState } from "react";
import { api } from "../api/client";
import DebateTranscript from "../components/DebateTranscript";

export default function DebateViewerPage() {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.debate().then(setRows).catch((e) => setError(String((e as Error).message ?? e)));
  }, []);

  return (
    <div className="stack">
      <div className="page-head">
        <div className="title">
          <h2>LangGraph Debate</h2>
          <p>9-node adversarial bull vs bear debate · gating decisions</p>
        </div>
        {rows.length > 0 && <span className="chip">{rows.length} decisions</span>}
      </div>

      {error && <div className="error-box">⚠ {error}</div>}
      {rows.length === 0 && !error && (
        <div className="empty-note">No debate cycles yet — run `python -m langgraph_app.src.graph_definition --mock`</div>
      )}
      {rows.map((r, i) => <DebateTranscript key={i} row={r} />)}
    </div>
  );
}
