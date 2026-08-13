import { useEffect, useState } from "react";
import { api } from "../api/client";
import DebateTranscript from "../components/DebateTranscript";

export default function DebateViewerPage() {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);

  useEffect(() => {
    api.debate().then(setRows).catch(() => setRows([]));
  }, []);

  return (
    <section>
      <h2>Bull / Bear Debate Viewer</h2>
      {rows.length === 0 && <p style={{ color: "#888" }}>No debate cycles yet — run `python -m langgraph_app.src.graph_definition --mock`</p>}
      {rows.map((r, i) => (
        <DebateTranscript key={i} row={r} />
      ))}
    </section>
  );
}
