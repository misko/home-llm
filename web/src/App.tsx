import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { ChatPage } from "./pages/ChatPage";
import { ModelsPage } from "./pages/ModelsPage";
import { StoragePage } from "./pages/StoragePage";
import { SystemPage } from "./pages/SystemPage";

export function App() {
  return (
    <BrowserRouter basename="/ui">
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<ChatPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="benchmarks" element={<BenchmarksPage />} />
          <Route path="storage" element={<StoragePage />} />
          <Route path="system" element={<SystemPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
