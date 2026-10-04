import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell";
import FaceToSketch from "./pages/FaceToSketch";
import HardRouting from "./pages/HardRouting";
import SoftMoE from "./pages/SoftMoE";
import Universal from "./pages/Universal";
import { Icon } from "./components/ui";

function NotFound() {
  return (
    <div className="card flex flex-col items-center gap-3 px-6 py-16 text-center">
      <Icon name="location_off" className="text-4xl text-outline" />
      <h1 className="font-headline text-xl font-semibold text-on-surface">Workspace not found</h1>
      <p className="max-w-md text-sm text-on-surface-variant">
        Pick one of the four task workspaces from the navigation bar above.
      </p>
      <Navigate to="/universal" replace />
    </div>
  );
}

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Navigate to="/universal" replace />} />
        <Route path="/universal" element={<Universal />} />
        <Route path="/hard-routing" element={<HardRouting />} />
        <Route path="/soft-moe" element={<SoftMoE />} />
        <Route path="/face-to-sketch" element={<FaceToSketch />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Shell>
  );
}
