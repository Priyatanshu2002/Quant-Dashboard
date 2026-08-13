import { NavLink, Route, Routes } from "react-router-dom";
import ScreenerPage from "./pages/screener";
import FinancialsPage from "./pages/financials";
import BacktestResultsPage from "./pages/backtest_results";
import PortfolioPage from "./pages/portfolio";
import DebateViewerPage from "./pages/debate_viewer";
import ValuationPage from "./pages/valuation";

const NAV = [
  { to: "/", label: "Screener", end: true },
  { to: "/financials", label: "Financials" },
  { to: "/valuation", label: "Valuation" },
  { to: "/backtest", label: "Backtests" },
  { to: "/portfolio", label: "Portfolio" },
  { to: "/debate", label: "Debate" },
];

export default function App() {
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 1100, margin: "0 auto", padding: 16 }}>
      <header style={{ display: "flex", alignItems: "center", gap: 24, borderBottom: "1px solid #ddd", paddingBottom: 12 }}>
        <h1 style={{ fontSize: 18, margin: 0 }}>⚔️ Agonistes</h1>
        <nav style={{ display: "flex", gap: 12 }}>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              style={({ isActive }) => ({
                textDecoration: "none",
                fontWeight: isActive ? 700 : 400,
                color: isActive ? "#1a5cff" : "#333",
              })}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main style={{ paddingTop: 16 }}>
        <Routes>
          <Route path="/" element={<ScreenerPage />} />
          <Route path="/financials" element={<FinancialsPage />} />
          <Route path="/valuation" element={<ValuationPage />} />
          <Route path="/backtest" element={<BacktestResultsPage />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
          <Route path="/debate" element={<DebateViewerPage />} />
        </Routes>
      </main>
    </div>
  );
}
