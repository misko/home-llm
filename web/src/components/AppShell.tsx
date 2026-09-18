import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  BarChart3,
  Boxes,
  Cpu,
  HardDrive,
  MessageSquareText,
  Settings2,
} from "lucide-react";
import { RuntimeBar } from "./RuntimeBar";
import { useConsoleEvents, useRuntime } from "../hooks/useConsoleData";
import { consoleApi } from "../api/client";
import { consoleKeys } from "../hooks/useConsoleData";
import { registerConsoleWebMcp } from "../webmcp";

const navigation = [
  ["Chat", "/", MessageSquareText],
  ["Models", "/models", Boxes],
  ["Benchmarks", "/benchmarks", BarChart3, "desktop-only"],
  ["Storage", "/storage", HardDrive],
  ["System", "/system", Settings2],
  ["Conversation", "/?settings=conversation", Settings2, "mobile-only"],
] as const;

export function AppShell() {
  const [railOpen, setRailOpen] = useState(false);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const runtime = useRuntime();
  useConsoleEvents();
  useEffect(() => registerConsoleWebMcp({
    api: consoleApi,
    navigate,
    refresh: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: consoleKeys.runtime }),
        queryClient.invalidateQueries({ queryKey: consoleKeys.portfolio }),
      ]);
    },
  }), [navigate, queryClient]);

  return (
    <div className="app-shell">
      <aside className={railOpen ? "rail open" : "rail"}>
        <button className="brand" aria-label="LLM Lab home" onClick={() => navigate("/")}>
          <span className="brand-mark"><Cpu size={19} /></span>
          <span>LLM LAB</span>
        </button>
        <nav aria-label="Primary navigation">
          {navigation.map(([label, path, Icon, visibility]) => (
            <NavLink
              to={path}
              end={path === "/"}
              className={({ isActive }) => `${isActive ? "nav-item active" : "nav-item"}${visibility ? ` ${visibility}` : ""}`}
              key={label}
              onClick={() => setRailOpen(false)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="rail-foot">
          <span className={runtime.data?.ready ? "signal-dot" : "signal-dot offline"} />
          {runtime.data?.ready ? "Gateway online" : "Gateway waiting"}
        </div>
      </aside>
      <main>
        <RuntimeBar onSwitch={() => navigate("/models")} />
        <Outlet />
      </main>
    </div>
  );
}
