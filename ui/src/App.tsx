import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity, Briefcase, Gauge, LineChart, MessageSquare, MonitorCog, Scale, Trophy, Waypoints, Zap,
} from "lucide-react";
import ScreenerPage from "./pages/screener";
import FinancialsPage from "./pages/financials";
import ValuationPage from "./pages/valuation";
import BacktestResultsPage from "./pages/backtest_results";
import PortfolioPage from "./pages/portfolio";
import DebateViewerPage from "./pages/debate_viewer";
import MonitoringPage from "./pages/monitoring";
import OnchainPage from "./pages/onchain";
import BenchmarkPage from "./pages/benchmark";

interface NavEntry { to: string; label: string; icon: React.ElementType; end?: boolean; }
const NAV: { group: string; items: NavEntry[] }[] = [
  {
    group: "Analytics",
    items: [
      { to: "/", label: "Screener", icon: Gauge, end: true },
      { to: "/onchain", label: "On-chain", icon: Waypoints },
      { to: "/financials", label: "Financials", icon: LineChart },
      { to: "/valuation", label: "Valuation", icon: Scale },
    ],
  },
  {
    group: "Engine",
    items: [
      { to: "/backtest", label: "Backtests", icon: Activity },
      { to: "/benchmark", label: "Benchmark", icon: Trophy },
      { to: "/portfolio", label: "Portfolio", icon: Briefcase },
      { to: "/debate", label: "Debate", icon: MessageSquare },
    ],
  },
  {
    group: "Operations",
    items: [
      { to: "/monitoring", label: "Monitoring", icon: MonitorCog },
    ],
  },
];

const TITLES: Record<string, string> = {
  "/": "Screener", "/onchain": "On-chain", "/financials": "Fundamentals", "/valuation": "Valuation",
  "/backtest": "Backtest Engine", "/benchmark": "Model Benchmark", "/portfolio": "Portfolio", "/debate": "LangGraph Debate",
  "/monitoring": "System Monitoring",
};

export default function App() {
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? "Agonistes";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo"><Zap size={18} /></div>
          <div>
            <div className="title">Agonistes</div>
            <div className="sub">Quant OS</div>
          </div>
        </div>

        {NAV.map((g) => (
          <div key={g.group}>
            <div className="nav-group">{g.group}</div>
            {g.items.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
                >
                  <Icon />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        ))}

        <div className="sidebar-foot">
          <div className="status"><span className="dot green" /> Systems nominal</div>
          <div className="status"><span className="dot gray" /> SQLite · dev</div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <h1>{title}</h1>
          <span className="crumb">Project Agonistes · autonomous quant trading system</span>
          <div className="right">
            <span className="status"><span className="dot green" /> Live</span>
          </div>
        </header>
        <main className="content">
          <Routes>
            <Route path="/" element={<ScreenerPage />} />
            <Route path="/onchain" element={<OnchainPage />} />
            <Route path="/financials" element={<FinancialsPage />} />
            <Route path="/financials/:symbol" element={<FinancialsPage />} />
            <Route path="/valuation" element={<ValuationPage />} />
            <Route path="/valuation/:symbol" element={<ValuationPage />} />
            <Route path="/backtest" element={<BacktestResultsPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/debate" element={<DebateViewerPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
