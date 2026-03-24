"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  // Gradient Descent 1D
  gdLoss, gdGradient, gdStep, gdCurvePoints, thetaToSvgX, lossToSvgY, gdTangentLine,
  // Gradient Descent 2D (3D surface)
  gdLoss2D, gdGradient2D, gdStep2D, gdConverged2D,
  GD2D_T1_RANGE, GD2D_T2_RANGE, GD2D_MIN, GD2D_LOSS_MIN, GD2D_LOSS_MAX,
  // K-Means
  generateClusteredPoints, initializeCentroids, kMeansStep, assignPoints,
  type Point2D, type KMeansState,
  // Linear Regression
  generateLinearData, computeMSE, linRegStep, type LinRegState,
  // Neural Network
  randomWeights, forwardPass, type NNWeights, type NNActivations, type ActivationFn,
} from "../lib/mlAlgorithms";

// ─── Color palette ────────────────────────────────────────────────────────────
const C = {
  primary:   "#7a6248",
  blue:      "#3b90cc",
  orange:    "#c86030",
  green:     "#3a9a50",
  purple:    "#a090c8",
  gold:      "#b89020",
} as const;

const CLUSTER_COLORS = [C.blue, C.purple, C.orange, C.green, C.gold];

// ─── Topic types ──────────────────────────────────────────────────────────────
type Topic = "gradient-descent" | "kmeans" | "linear-regression" | "neural-network";

const TOPICS: { id: Topic; label: string; short: string }[] = [
  { id: "gradient-descent",   label: "Gradient Descent",     short: "GD" },
  { id: "kmeans",             label: "K-Means Clustering",   short: "KM" },
  { id: "linear-regression",  label: "Linear Regression",    short: "LR" },
  { id: "neural-network",     label: "Neural Network",        short: "NN" },
];

// ─── Shared button styles (mirrors SortingVisualizer) ─────────────────────────
function primaryBtn(active: boolean, running: boolean, disabled: boolean) {
  const base = "flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200 border";
  if (disabled) return `${base} bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] opacity-35 cursor-not-allowed`;
  if (active && running) return `${base} bg-[#c86030]/10 border-[#c86030]/30 text-[#c86030] hover:bg-[#c86030]/18`;
  if (active) return `${base} bg-[#7a6248]/10 border-[#7a6248]/30 text-[#7a6248] hover:bg-[#7a6248]/18 shadow-sm`;
  return `${base} bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]`;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">
      {children}
    </p>
  );
}

function StatCard({ label, value, color = C.primary }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">{label}</span>
      <span className="font-mono text-sm font-semibold" style={{ color }}>{value}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GRADIENT DESCENT — shared state hook
// ─────────────────────────────────────────────────────────────────────────────
const GD2D_START: [number, number] = [-1.5, 4.0];

function useGradientDescentState() {
  // ── 1D state ──────────────────────────────────────────────────────────────
  const [theta, setTheta]           = useState(-3.5);
  const [lr, setLr]                 = useState(0.15);
  const [startTheta, setStartTheta] = useState(-3.5);
  const [running, setRunning]       = useState(false);
  const [iteration, setIteration]   = useState(0);
  const [history, setHistory]       = useState<number[]>([-3.5]);

  // ── 3D state ──────────────────────────────────────────────────────────────
  const [view3d, setView3d]       = useState(false);
  const [pos2d, setPos2d]         = useState<[number, number]>(GD2D_START);
  const [history2d, setHistory2d] = useState<[number, number][]>([GD2D_START]);
  const [azimuth, setAzimuth]     = useState(Math.PI / 5);

  // ── Mutable refs (avoid stale closures in loops) ──────────────────────────
  const runRef        = useRef(false);
  const timerRef      = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thetaRef      = useRef(theta);
  const lrRef         = useRef(lr);
  const view3dRef     = useRef(false);
  const pos2dRef      = useRef<[number, number]>(GD2D_START);
  const startThetaRef = useRef(-3.5);
  thetaRef.current      = theta;
  lrRef.current         = lr;
  view3dRef.current     = view3d;
  pos2dRef.current      = pos2d;
  startThetaRef.current = startTheta;

  const doStep = useCallback(() => {
    if (view3dRef.current) {
      const next = gdStep2D(pos2dRef.current, lrRef.current);
      setPos2d(next);
      setHistory2d((h) => [...h.slice(-80), next]);
      setIteration((i) => i + 1);
    } else {
      const next = gdStep(thetaRef.current, lrRef.current);
      setTheta(next);
      setHistory((h) => [...h.slice(-60), next]);
      setIteration((i) => i + 1);
    }
  }, []);

  const runLoop = useCallback(() => {
    if (!runRef.current) return;
    if (view3dRef.current) {
      const next = gdStep2D(pos2dRef.current, lrRef.current);
      const conv = gdConverged2D(next);
      setPos2d(next);
      pos2dRef.current = next;
      setHistory2d((prev) => [...prev.slice(-80), next]);
      setIteration((i) => i + 1);
      if (conv) { runRef.current = false; setRunning(false); return; }
      timerRef.current = setTimeout(runLoop, 60);
    } else {
      const next = gdStep(thetaRef.current, lrRef.current);
      const conv = Math.abs(gdGradient(next)) < 0.001;
      setTheta(next);
      setHistory((prev) => [...prev.slice(-60), next]);
      setIteration((i) => i + 1);
      if (conv) { runRef.current = false; setRunning(false); return; }
      timerRef.current = setTimeout(runLoop, 60);
    }
  }, []);

  const handleRun = useCallback(() => {
    if (running) {
      runRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      setRunning(false);
    } else {
      const notConverged = view3dRef.current
        ? !gdConverged2D(pos2dRef.current)
        : Math.abs(gdGradient(thetaRef.current)) >= 0.01;
      if (notConverged) { runRef.current = true; setRunning(true); runLoop(); }
    }
  }, [running, runLoop]);

  const handleReset = useCallback(() => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    setIteration(0);
    if (view3dRef.current) {
      setPos2d(GD2D_START);
      pos2dRef.current = GD2D_START;
      setHistory2d([GD2D_START]);
    } else {
      setTheta(startThetaRef.current);
      setHistory([startThetaRef.current]);
    }
  }, []);

  const switchView = useCallback((to3d: boolean) => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    setIteration(0);
    setView3d(to3d);
    if (to3d) {
      setPos2d(GD2D_START);
      pos2dRef.current = GD2D_START;
      setHistory2d([GD2D_START]);
    } else {
      setTheta(startThetaRef.current);
      setHistory([startThetaRef.current]);
    }
  }, []);

  useEffect(() => () => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  return {
    theta, setTheta, lr, setLr, startTheta, setStartTheta,
    running, iteration, history, setHistory, setIteration,
    doStep, handleRun, handleReset,
    view3d, switchView, pos2d, history2d, azimuth, setAzimuth,
  };
}

type GDState = ReturnType<typeof useGradientDescentState>;

// ─── Gradient Descent Viz ─────────────────────────────────────────────────────
function GradientDescentViz({ s }: { s: GDState }) {
  const SVG_W = 500, SVG_H = 280, PAD = 36;

  const loss     = gdLoss(s.theta);
  const gradient = gdGradient(s.theta);
  const converged = Math.abs(gradient) < 0.01;
  const curvePoints = gdCurvePoints(SVG_W, SVG_H, PAD);
  const pathD = curvePoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  const ptX = thetaToSvgX(s.theta, SVG_W, PAD);
  const ptY = lossToSvgY(loss, SVG_H, PAD);
  const tangent = gdTangentLine(s.theta, SVG_W, SVG_H, PAD);

  return (
    <>
      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-full" style={{ maxHeight: "100%" }}>
        {/* Axes */}
        <line x1={PAD} y1={PAD} x2={PAD} y2={SVG_H - PAD} stroke="#ddddd4" strokeWidth="1.5" />
        <line x1={PAD} y1={SVG_H - PAD} x2={SVG_W - PAD} y2={SVG_H - PAD} stroke="#ddddd4" strokeWidth="1.5" />
        {/* Axis labels */}
        <text x={SVG_W / 2} y={SVG_H - 4} textAnchor="middle" fontSize="10" fill="#6b6b60">θ (parameter)</text>
        <text x={10} y={SVG_H / 2} textAnchor="middle" fontSize="10" fill="#6b6b60" transform={`rotate(-90, 10, ${SVG_H / 2})`}>L(θ)</text>
        {/* Loss curve */}
        <path d={pathD} fill="none" stroke={C.blue} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        {/* History trail — small dots along the curve */}
        {s.history.map((t, i) => {
          const hx = thetaToSvgX(t, SVG_W, PAD);
          const hy = lossToSvgY(gdLoss(t), SVG_H, PAD);
          return (
            <circle key={i} cx={hx} cy={hy} r={2.5}
              fill={C.orange} opacity={0.2 + (i / s.history.length) * 0.65} />
          );
        })}
        {/* Tangent line */}
        <line x1={tangent.x1} y1={tangent.y1} x2={tangent.x2} y2={tangent.y2}
          stroke={C.gold} strokeWidth="1.5" strokeDasharray="5 4" opacity={0.85} />
        {/* Current point */}
        <motion.circle
          cx={ptX} cy={ptY} r={8}
          fill={C.orange} stroke="white" strokeWidth="2.5"
          animate={{ cx: ptX, cy: ptY }}
          transition={{ type: "spring", stiffness: 320, damping: 28 }}
        />
        {/* Minimum marker */}
        <circle cx={thetaToSvgX(2, SVG_W, PAD)} cy={lossToSvgY(1, SVG_H, PAD)}
          r={4} fill="none" stroke={C.green} strokeWidth="1.5" strokeDasharray="3 2" opacity={0.7} />
        <text x={thetaToSvgX(2, SVG_W, PAD) + 8} y={lossToSvgY(1, SVG_H, PAD) + 4}
          fontSize="9" fill={C.green} opacity={0.8}>min</text>
      </svg>
      <div className="absolute bottom-3 right-4 flex gap-4 text-[10px] text-[#6b6b60] pointer-events-none">
        <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-0.5 rounded" style={{ background: C.blue }} />L(θ) curve</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: C.orange }} />current θ</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-px border-t-2 border-dashed" style={{ borderColor: C.gold }} />tangent</span>
      </div>
      {converged && (
        <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-[#ddddd4] shadow-sm text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: C.green }} />
          <span className="font-medium" style={{ color: C.green }}>Converged</span>
        </div>
      )}
      {!s.running && !converged && s.iteration === 0 && (
        <p className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] tracking-widest uppercase text-[#6b6b60]/50 pointer-events-none select-none">
          Press Step or Run to begin
        </p>
      )}
    </>
  );
}

// ─── Gradient Descent 3D Surface Viz ─────────────────────────────────────────
function GradientDescentViz3D({ s }: { s: GDState }) {
  const svgRef  = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<{ startX: number; startAz: number } | null>(null);

  // Perspective projection: orthographic + rotation/elevation
  function project(t1: number, t2: number, loss: number, azimuth: number, svgW: number, svgH: number): [number, number, number] {
    const elev = 0.52; // ~30 degrees
    const scale = Math.min(svgW, svgH) * 0.38;
    // Normalise to [-1, 1] range
    const nx = (t1 - (GD2D_T1_RANGE[0] + GD2D_T1_RANGE[1]) / 2) / ((GD2D_T1_RANGE[1] - GD2D_T1_RANGE[0]) / 2);
    const ny = (t2 - (GD2D_T2_RANGE[0] + GD2D_T2_RANGE[1]) / 2) / ((GD2D_T2_RANGE[1] - GD2D_T2_RANGE[0]) / 2);
    const nz = (loss - GD2D_LOSS_MIN) / (GD2D_LOSS_MAX - GD2D_LOSS_MIN);
    // Rotate around vertical axis by azimuth
    const rx = nx * Math.cos(azimuth) - ny * Math.sin(azimuth);
    const ry = nx * Math.sin(azimuth) + ny * Math.cos(azimuth);
    const rz = nz;
    // Apply elevation tilt
    const sx = rx;
    const sy = -rz * Math.sin(elev) - ry * Math.cos(elev);
    const depth = ry * Math.sin(elev) - rz * Math.cos(elev);
    return [
      svgW / 2 + sx * scale,
      svgH / 2 + sy * scale * 0.9 - 10,
      depth,
    ];
  }

  function lossToColor(loss: number): string {
    const t = Math.min(1, Math.max(0, (loss - GD2D_LOSS_MIN) / (GD2D_LOSS_MAX - GD2D_LOSS_MIN)));
    // Blue → gold → orange
    if (t < 0.5) {
      const u = t * 2;
      const r = Math.round(59  + (184 - 59)  * u);
      const g = Math.round(144 + (144 - 144) * u);
      const b = Math.round(204 + (32  - 204) * u);
      return `rgb(${r},${g},${b})`;
    } else {
      const u = (t - 0.5) * 2;
      const r = Math.round(184 + (200 - 184) * u);
      const g = Math.round(144 + (96  - 144) * u);
      const b = Math.round(32  + (48  - 32)  * u);
      return `rgb(${r},${g},${b})`;
    }
  }

  const SVG_W = 520, SVG_H = 320;
  const GRID = 18;

  // Build quads
  type Quad = { points: [number, number][]; fill: string; depth: number };
  const quads: Quad[] = [];
  const t1Step = (GD2D_T1_RANGE[1] - GD2D_T1_RANGE[0]) / GRID;
  const t2Step = (GD2D_T2_RANGE[1] - GD2D_T2_RANGE[0]) / GRID;
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const corners: [number, number, number][] = [
        [GD2D_T1_RANGE[0] + i * t1Step,       GD2D_T2_RANGE[0] + j * t2Step,       gdLoss2D(GD2D_T1_RANGE[0] + i * t1Step,       GD2D_T2_RANGE[0] + j * t2Step)],
        [GD2D_T1_RANGE[0] + (i+1) * t1Step,   GD2D_T2_RANGE[0] + j * t2Step,       gdLoss2D(GD2D_T1_RANGE[0] + (i+1) * t1Step,   GD2D_T2_RANGE[0] + j * t2Step)],
        [GD2D_T1_RANGE[0] + (i+1) * t1Step,   GD2D_T2_RANGE[0] + (j+1) * t2Step,   gdLoss2D(GD2D_T1_RANGE[0] + (i+1) * t1Step,   GD2D_T2_RANGE[0] + (j+1) * t2Step)],
        [GD2D_T1_RANGE[0] + i * t1Step,       GD2D_T2_RANGE[0] + (j+1) * t2Step,   gdLoss2D(GD2D_T1_RANGE[0] + i * t1Step,       GD2D_T2_RANGE[0] + (j+1) * t2Step)],
      ];
      const projected = corners.map(([t1, t2, l]) => project(t1, t2, l, s.azimuth, SVG_W, SVG_H));
      const avgDepth  = projected.reduce((acc, p) => acc + p[2], 0) / 4;
      const avgLoss   = corners.reduce((acc, c) => acc + c[2], 0) / 4;
      quads.push({
        points: projected.map(p => [p[0], p[1]] as [number, number]),
        fill: lossToColor(avgLoss),
        depth: avgDepth,
      });
    }
  }
  quads.sort((a, b) => a.depth - b.depth); // painter's algorithm

  // Project history path
  const pathPoints = s.history2d.map(([t1, t2]) => {
    const loss = gdLoss2D(t1, t2);
    return project(t1, t2, loss, s.azimuth, SVG_W, SVG_H);
  });

  // Current position
  const [cx, cy] = (() => {
    const [t1, t2] = s.pos2d;
    const loss = gdLoss2D(t1, t2);
    const p = project(t1, t2, loss, s.azimuth, SVG_W, SVG_H);
    return [p[0], p[1]];
  })();

  // Minimum marker
  const [minX, minY] = (() => {
    const p = project(GD2D_MIN[0], GD2D_MIN[1], GD2D_LOSS_MIN + 0.05, s.azimuth, SVG_W, SVG_H);
    return [p[0], p[1]];
  })();

  const converged = gdConverged2D(s.pos2d);

  // Drag to rotate
  function onMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    dragRef.current = { startX: e.clientX, startAz: s.azimuth };
  }
  function onMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    s.setAzimuth(dragRef.current.startAz + dx * 0.012);
  }
  function onMouseUp() { dragRef.current = null; }

  return (
    <>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        style={{ maxHeight: "100%", userSelect: "none" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {/* Surface quads */}
        {quads.map((q, qi) => (
          <polygon
            key={qi}
            points={q.points.map(p => p.join(",")).join(" ")}
            fill={q.fill}
            stroke="rgba(255,255,255,0.18)"
            strokeWidth="0.5"
          />
        ))}
        {/* History path on surface */}
        {pathPoints.length > 1 && (
          <polyline
            points={pathPoints.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ")}
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.9}
          />
        )}
        {/* History dots */}
        {pathPoints.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={2}
            fill="white" opacity={0.3 + (i / pathPoints.length) * 0.6} />
        ))}
        {/* Minimum marker */}
        <circle cx={minX} cy={minY} r={5} fill="none" stroke={C.green} strokeWidth="1.5" opacity={0.85} />
        <text x={minX + 7} y={minY + 4} fontSize="9" fill={C.green} opacity={0.85}>min</text>
        {/* Current position */}
        <circle cx={cx} cy={cy} r={8} fill={C.orange} stroke="white" strokeWidth="2.5" />
      </svg>

      {/* Drag hint */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-[10px] text-[#6b6b60]/60 pointer-events-none select-none">
        drag to rotate
      </div>

      {converged && (
        <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-[#ddddd4] shadow-sm text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: C.green }} />
          <span className="font-medium" style={{ color: C.green }}>Converged</span>
        </div>
      )}
      {!s.running && !converged && s.iteration === 0 && (
        <p className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] tracking-widest uppercase text-[#6b6b60]/50 pointer-events-none select-none">
          Press Step or Run to begin
        </p>
      )}
    </>
  );
}

// ─── Gradient Descent Sidebar ─────────────────────────────────────────────────
function GradientDescentSidebar({ s }: { s: GDState }) {
  const loss1d   = gdLoss(s.theta);
  const gradient = gdGradient(s.theta);
  const converged1d = Math.abs(gradient) < 0.01;
  const converged2d = gdConverged2D(s.pos2d);
  const converged   = s.view3d ? converged2d : converged1d;
  const loss2d = gdLoss2D(s.pos2d[0], s.pos2d[1]);
  const [g1, g2] = gdGradient2D(s.pos2d[0], s.pos2d[1]);

  return (
    <div className="p-4 sm:p-6 space-y-6">
      <div>
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Gradient Descent</h2>
        <div className="w-10 h-px mb-3" style={{ background: C.primary }} />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          {s.view3d
            ? <>Minimizing L(θ₁,θ₂) on a 2D surface. Update: <span className="font-mono" style={{ color: C.blue }}>θ ← θ − α·∇L</span></>
            : <>Iterative optimization that minimizes L(θ) by stepping opposite to the gradient.
               Update rule: <span className="font-mono" style={{ color: C.blue }}>θ ← θ − α·∇L(θ)</span></>
          }
        </p>
      </div>

      {/* 2D / 3D toggle */}
      <div>
        <SectionLabel>View</SectionLabel>
        <div className="flex gap-2">
          <button
            onClick={() => s.switchView(false)}
            className={`flex-1 py-2 rounded-lg text-xs font-medium border transition-all duration-200 ${
              !s.view3d
                ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248]"
                : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:text-[#1a1a1a]"
            }`}
          >
            1D curve
          </button>
          <button
            onClick={() => s.switchView(true)}
            className={`flex-1 py-2 rounded-lg text-xs font-medium border transition-all duration-200 ${
              s.view3d
                ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248]"
                : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:text-[#1a1a1a]"
            }`}
          >
            3D surface
          </button>
        </div>
      </div>

      <div>
        <SectionLabel>Live Stats</SectionLabel>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 grid grid-cols-2 gap-3">
          {s.view3d ? (
            <>
              <StatCard label="θ₁" value={s.pos2d[0].toFixed(3)} color={C.orange} />
              <StatCard label="θ₂" value={s.pos2d[1].toFixed(3)} color={C.purple} />
              <StatCard label="L(θ)" value={loss2d.toFixed(4)} color={C.blue} />
              <StatCard label="‖∇L‖" value={Math.sqrt(g1*g1+g2*g2).toFixed(4)} color={C.gold} />
            </>
          ) : (
            <>
              <StatCard label="θ" value={s.theta.toFixed(4)} color={C.orange} />
              <StatCard label="L(θ)" value={loss1d.toFixed(4)} color={C.blue} />
              <StatCard label="∇L(θ)" value={gradient.toFixed(4)} color={C.gold} />
              <StatCard label="Iteration" value={String(s.iteration)} color={C.primary} />
            </>
          )}
          {s.view3d && <StatCard label="Iteration" value={String(s.iteration)} color={C.primary} />}
        </div>
      </div>
      <div className="space-y-4">
        <SectionLabel>Controls</SectionLabel>
        <div>
          <div className="flex justify-between mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">Learning Rate α</span>
            <span className="text-xs font-mono font-semibold" style={{ color: C.primary }}>{s.lr.toFixed(2)}</span>
          </div>
          <input type="range" min={1} max={100} value={Math.round(s.lr * 100)}
            onChange={(e) => { if (!s.running) s.setLr(Number(e.target.value) / 100); }}
            disabled={s.running} className="w-full" />
          <div className="flex justify-between text-[9px] text-[#6b6b60] mt-1"><span>0.01</span><span>1.0</span></div>
        </div>
        {!s.view3d && (
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">Start Position θ₀</span>
              <span className="text-xs font-mono font-semibold" style={{ color: C.primary }}>{s.startTheta.toFixed(1)}</span>
            </div>
            <input type="range" min={-40} max={79} value={Math.round(s.startTheta * 10)}
              onChange={(e) => {
                if (!s.running) {
                  const v = Number(e.target.value) / 10;
                  s.setStartTheta(v);
                  s.setTheta(v);
                  s.setHistory([v]);
                  s.setIteration(0);
                }
              }}
              disabled={s.running} className="w-full" />
            <div className="flex justify-between text-[9px] text-[#6b6b60] mt-1"><span>−4</span><span>7.9</span></div>
          </div>
        )}
      </div>
      <div className="flex gap-2">
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.doStep}
          disabled={s.running || converged}
          className={primaryBtn(false, false, s.running || converged)}>Step</motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleRun}
          className={primaryBtn(true, s.running, converged && !s.running)}>
          {s.running ? "Pause" : converged ? "Done ✓" : "Run"}
        </motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleReset}
          className={primaryBtn(false, false, false)}>Reset</motion.button>
      </div>
      <div>
        <SectionLabel>Formula</SectionLabel>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 font-mono text-xs space-y-1 text-center">
          {s.view3d ? (
            <>
              <div style={{ color: C.blue }}>L(θ) = (θ₁−2)² + 0.8(θ₂−1.5)² + 0.5</div>
              <div className="text-[#6b6b60]">∇L = [2(θ₁−2), 1.6(θ₂−1.5)]</div>
              <div className="text-[#6b6b60]">θ ← θ − α · ∇L</div>
            </>
          ) : (
            <>
              <div style={{ color: C.blue }}>L(θ) = (θ − 2)² + 1</div>
              <div className="text-[#6b6b60]">∇L(θ) = 2(θ − 2)</div>
              <div className="text-[#6b6b60]">θ ← θ − α · ∇L(θ)</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// K-MEANS CLUSTERING — shared state hook
// ─────────────────────────────────────────────────────────────────────────────
function useKMeansState() {
  const [points, setPoints]           = useState<Point2D[]>(() => generateClusteredPoints(40));
  const [k, setK]                     = useState(3);
  const [centroids, setCentroids]     = useState<Point2D[]>([]);
  const [assignments, setAssignments] = useState<number[]>([]);
  const [iteration, setIteration]     = useState(0);
  const [converged, setConverged]     = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [running, setRunning]         = useState(false);

  const runRef   = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stateRef = useRef<KMeansState | null>(null);

  const handleInit = useCallback(() => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    // Read current points from stateRef to avoid stale closure
    const pts = stateRef.current?.points ?? points;
    const cents = initializeCentroids(pts, k);
    const asgn  = assignPoints(pts, cents);
    setCentroids(cents);
    setAssignments(asgn);
    setIteration(0);
    setConverged(false);
    setInitialized(true);
    stateRef.current = { points: pts, centroids: cents, assignments: asgn, k, converged: false, iteration: 0 };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, points]);

  const handleStep = useCallback(() => {
    if (!stateRef.current || stateRef.current.converged) return;
    const next = kMeansStep(stateRef.current);
    stateRef.current = next;
    setCentroids([...next.centroids]);
    setAssignments([...next.assignments]);
    setIteration(next.iteration);
    setConverged(next.converged);
  }, []);

  const runLoop = useCallback(() => {
    if (!runRef.current || !stateRef.current) return;
    if (stateRef.current.converged) {
      runRef.current = false;
      setRunning(false);
      return;
    }
    const next = kMeansStep(stateRef.current);
    stateRef.current = next;
    setCentroids([...next.centroids]);
    setAssignments([...next.assignments]);
    setIteration(next.iteration);
    setConverged(next.converged);
    if (!next.converged) {
      timerRef.current = setTimeout(runLoop, 300);
    } else {
      runRef.current = false;
      setRunning(false);
    }
  }, []);

  const handleRun = useCallback(() => {
    if (!initialized) return;
    if (running) {
      runRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      setRunning(false);
    } else if (!converged) {
      runRef.current = true;
      setRunning(true);
      runLoop();
    }
  }, [initialized, running, converged, runLoop]);

  const handleNewData = useCallback(() => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    const newPts = generateClusteredPoints(40);
    setPoints(newPts);
    setCentroids([]);
    setAssignments([]);
    setIteration(0);
    setConverged(false);
    setInitialized(false);
    stateRef.current = null;
  }, []);

  useEffect(() => {
    return () => {
      runRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Re-init when k changes (reset centroids so stale k-count centroids don't persist)
  useEffect(() => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    setCentroids([]);
    setAssignments([]);
    setIteration(0);
    setConverged(false);
    setInitialized(false);
    stateRef.current = null;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k]);

  // Keep stateRef.points in sync when points change
  useEffect(() => {
    if (stateRef.current) {
      stateRef.current = { ...stateRef.current, points };
    }
  }, [points]);

  return {
    points, k, setK, centroids, assignments, iteration, converged, initialized, running,
    handleInit, handleStep, handleRun, handleNewData,
  };
}

type KMState = ReturnType<typeof useKMeansState>;

function KMeansViz({ s }: { s: KMState }) {
  const W = 480, H = 320, PAD = 20;
  const toSvgX = (v: number) => PAD + v * (W - PAD * 2);
  const toSvgY = (v: number) => PAD + (1 - v) * (H - PAD * 2);

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" style={{ maxHeight: "100%" }}>
        {/* Points */}
        {s.points.map((p, i) => {
          const color = s.initialized && s.assignments[i] !== undefined
            ? CLUSTER_COLORS[s.assignments[i]]
            : "#a090c8";
          return (
            <motion.circle key={i}
              cx={toSvgX(p.x)} cy={toSvgY(p.y)} r={5}
              fill={color} opacity={0.75}
              animate={{ fill: color }}
              transition={{ duration: 0.35 }}
            />
          );
        })}
        {/* Centroids */}
        {s.centroids.map((c, i) => {
          const cx = toSvgX(c.x);
          const cy = toSvgY(c.y);
          const color = CLUSTER_COLORS[i];
          const size = 10;
          return (
            <motion.g key={i}
              animate={{ x: cx, y: cy }}
              transition={{ type: "spring", stiffness: 250, damping: 22 }}>
              <line x1={-size} y1={-size} x2={size} y2={size} stroke={color} strokeWidth="3" strokeLinecap="round" />
              <line x1={size} y1={-size} x2={-size} y2={size} stroke={color} strokeWidth="3" strokeLinecap="round" />
              <circle r={size + 2} fill="none" stroke={color} strokeWidth="1.5" opacity={0.4} />
            </motion.g>
          );
        })}
      </svg>
      {!s.initialized && (
        <p className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] tracking-widest uppercase text-[#6b6b60]/50 pointer-events-none select-none whitespace-nowrap">
          Press Initialize to place centroids
        </p>
      )}
      {s.converged && (
        <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-[#ddddd4] shadow-sm text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: C.green }} />
          <span className="font-medium" style={{ color: C.green }}>Converged</span>
        </div>
      )}
      {/* Legend */}
      <div className="absolute bottom-3 left-4 flex gap-3 text-[10px] text-[#6b6b60] pointer-events-none flex-wrap">
        {Array.from({ length: s.k }, (_, i) => (
          <span key={i} className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: CLUSTER_COLORS[i] }} />
            Cluster {i + 1}
          </span>
        ))}
      </div>
    </>
  );
}

function KMeansSidebar({ s }: { s: KMState }) {
  return (
    <div className="p-4 sm:p-6 space-y-6">
      <div>
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">K-Means Clustering</h2>
        <div className="w-10 h-px mb-3" style={{ background: C.primary }} />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          Partitions data into K clusters by alternating between assigning points to the nearest
          centroid (×) and updating centroid positions to the cluster mean.
        </p>
      </div>
      <div>
        <SectionLabel>Stats</SectionLabel>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 grid grid-cols-2 gap-3">
          <StatCard label="K" value={String(s.k)} color={C.primary} />
          <StatCard label="Iteration" value={String(s.iteration)} color={C.blue} />
          <StatCard label="Points" value="40" color={C.purple} />
          <StatCard label="Status" value={s.converged ? "Converged" : s.initialized ? "Running" : "Init"} color={s.converged ? C.green : C.gold} />
        </div>
      </div>
      <div>
        <SectionLabel>Controls</SectionLabel>
        <div>
          <div className="flex justify-between mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">K (clusters)</span>
            <span className="text-xs font-mono font-semibold" style={{ color: C.primary }}>{s.k}</span>
          </div>
          <input type="range" min={2} max={5} value={s.k}
            onChange={(e) => { if (!s.running) s.setK(Number(e.target.value)); }}
            disabled={s.running} className="w-full" />
          <div className="flex justify-between text-[9px] text-[#6b6b60] mt-1"><span>2</span><span>5</span></div>
        </div>
      </div>
      <div className="flex gap-2 flex-wrap">
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleInit}
          disabled={s.running}
          className={primaryBtn(false, false, s.running)}>Init</motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleStep}
          disabled={!s.initialized || s.converged || s.running}
          className={primaryBtn(false, false, !s.initialized || s.converged || s.running)}>Step</motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleRun}
          disabled={!s.initialized}
          className={primaryBtn(true, s.running, !s.initialized)}>
          {s.running ? "Pause" : s.converged ? "Done ✓" : "Run"}
        </motion.button>
      </div>
      <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleNewData}
        disabled={s.running}
        className={`w-full py-2.5 rounded-xl text-sm font-semibold border transition-all duration-200 ${s.running ? "opacity-35 cursor-not-allowed" : ""} bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]`}>
        New Data
      </motion.button>
      <div>
        <SectionLabel>Algorithm Steps</SectionLabel>
        <ol className="space-y-2">
          {["Place K centroids randomly (K-Means++)", "Assign each point to nearest centroid", "Move each centroid to its cluster mean", "Repeat until centroids stop moving"].map((step, i) => (
            <li key={i} className="flex gap-3 text-sm text-[#6b6b60]">
              <span className="flex-shrink-0 w-5 h-5 rounded-full bg-[#7a6248]/10 border border-[#7a6248]/25 text-[#7a6248] flex items-center justify-center text-[10px] font-bold">{i + 1}</span>
              <span className="leading-relaxed">{step}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// LINEAR REGRESSION — shared state hook
// ─────────────────────────────────────────────────────────────────────────────
function useLinRegState() {
  const initPoints = () => generateLinearData(22);
  const [state, setState] = useState<LinRegState>(() => {
    const pts = initPoints();
    return { points: pts, slope: 0, intercept: 0.5, mse: computeMSE(pts, 0, 0.5), iteration: 0, converged: false };
  });
  const [lr, setLr]       = useState(0.18);
  const [running, setRunning] = useState(false);

  const runRef   = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // lrRef so the loop always reads the latest lr
  const lrRef    = useRef(lr);
  lrRef.current  = lr;

  const doStep = useCallback(() => {
    setState((prev) => linRegStep(prev, lrRef.current));
  }, []);

  const runLoop = useCallback(() => {
    if (!runRef.current) return;
    setState((prev) => {
      const next = linRegStep(prev, lrRef.current);
      if (next.converged || next.iteration > 2000) {
        runRef.current = false;
        // Schedule outside the setter to avoid nesting state updates
        setTimeout(() => setRunning(false), 0);
      } else {
        timerRef.current = setTimeout(runLoop, 30);
      }
      return next;
    });
  }, []);

  const handleRun = useCallback(() => {
    if (running) {
      runRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      setRunning(false);
    } else {
      setState((prev) => {
        if (!prev.converged && prev.iteration < 2000) {
          runRef.current = true;
          setTimeout(() => setRunning(true), 0);
          timerRef.current = setTimeout(runLoop, 0);
        }
        return prev;
      });
    }
  }, [running, runLoop]);

  const handleReset = useCallback(() => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    setState((prev) => ({
      ...prev,
      slope: 0, intercept: 0.5,
      mse: computeMSE(prev.points, 0, 0.5),
      iteration: 0, converged: false,
    }));
  }, []);

  const handleNewData = useCallback(() => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    const pts = generateLinearData(22);
    setState({ points: pts, slope: 0, intercept: 0.5, mse: computeMSE(pts, 0, 0.5), iteration: 0, converged: false });
  }, []);

  useEffect(() => () => {
    runRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  return { state, lr, setLr, running, doStep, handleRun, handleReset, handleNewData };
}

type LRState = ReturnType<typeof useLinRegState>;

function LinearRegressionViz({ s }: { s: LRState }) {
  const W = 480, H = 320, PAD = 24;
  const toSvgX = (v: number) => PAD + v * (W - PAD * 2);
  const toSvgY = (v: number) => PAD + (1 - v) * (H - PAD * 2);

  // Extend regression line to full SVG width (x=0..1 in data space)
  const x1data = 0, x2data = 1;
  const y1data = s.state.slope * x1data + s.state.intercept;
  const y2data = s.state.slope * x2data + s.state.intercept;

  const svgX1 = toSvgX(x1data);
  const svgX2 = toSvgX(x2data);
  const svgY1 = toSvgY(y1data);
  const svgY2 = toSvgY(y2data);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" style={{ maxHeight: "100%" }}>
      {/* Axes */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#ddddd4" strokeWidth="1.5" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#ddddd4" strokeWidth="1.5" />
      {/* Residuals */}
      {s.state.points.map((p, i) => {
        const predY = s.state.slope * p.x + s.state.intercept;
        return (
          <line key={i}
            x1={toSvgX(p.x)} y1={toSvgY(p.y)}
            x2={toSvgX(p.x)} y2={toSvgY(predY)}
            stroke={C.orange} strokeWidth="1" opacity={0.25} />
        );
      })}
      {/* Data points (drawn on top of residuals) */}
      {s.state.points.map((p, i) => (
        <circle key={i} cx={toSvgX(p.x)} cy={toSvgY(p.y)} r={5}
          fill={C.blue} opacity={0.65} />
      ))}
      {/* Regression line — animated, spans full SVG width */}
      <motion.line
        x1={svgX1} y1={svgY1}
        x2={svgX2} y2={svgY2}
        stroke={C.orange} strokeWidth="2.5" strokeLinecap="round"
        animate={{ y1: svgY1, y2: svgY2, x1: svgX1, x2: svgX2 }}
        transition={{ duration: 0.08 }}
      />
    </svg>
  );
}

function LinearRegressionSidebar({ s }: { s: LRState }) {
  const done = s.state.converged || s.state.iteration >= 2000;

  return (
    <div className="p-4 sm:p-6 space-y-6">
      <div>
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Linear Regression</h2>
        <div className="w-10 h-px mb-3" style={{ background: C.primary }} />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          Fits a line ŷ = mx + b to data by minimizing Mean Squared Error using gradient descent.
          Watch the line converge to the best fit.
        </p>
      </div>
      <div>
        <SectionLabel>Live Stats</SectionLabel>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 grid grid-cols-2 gap-3">
          <StatCard label="Slope (m)" value={s.state.slope.toFixed(4)} color={C.orange} />
          <StatCard label="Intercept (b)" value={s.state.intercept.toFixed(4)} color={C.blue} />
          <StatCard label="MSE Loss" value={s.state.mse.toFixed(5)} color={C.gold} />
          <StatCard label="Iteration" value={String(s.state.iteration)} color={C.primary} />
        </div>
      </div>
      <div>
        <SectionLabel>Controls</SectionLabel>
        <div>
          <div className="flex justify-between mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">Learning Rate</span>
            <span className="text-xs font-mono font-semibold" style={{ color: C.primary }}>{s.lr.toFixed(2)}</span>
          </div>
          <input type="range" min={1} max={50} value={Math.round(s.lr * 100)}
            onChange={(e) => { if (!s.running) s.setLr(Number(e.target.value) / 100); }}
            disabled={s.running} className="w-full" />
          <div className="flex justify-between text-[9px] text-[#6b6b60] mt-1"><span>0.01</span><span>0.50</span></div>
        </div>
      </div>
      <div className="flex gap-2">
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.doStep}
          disabled={s.running || done}
          className={primaryBtn(false, false, s.running || done)}>Step</motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleRun}
          className={primaryBtn(true, s.running, done && !s.running)}>
          {s.running ? "Pause" : done ? "Done ✓" : "Run"}
        </motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleReset}
          className={primaryBtn(false, false, false)}>Reset</motion.button>
      </div>
      <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleNewData}
        disabled={s.running}
        className={`w-full py-2.5 rounded-xl text-sm font-semibold border transition-all duration-200 ${s.running ? "opacity-35 cursor-not-allowed" : ""} bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]`}>
        New Data
      </motion.button>
      <div>
        <SectionLabel>Objective</SectionLabel>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 font-mono text-xs space-y-1 text-center">
          <div style={{ color: C.orange }}>ŷ = m·x + b</div>
          <div className="text-[#6b6b60]">MSE = (1/n) Σ (ŷᵢ − yᵢ)²</div>
          <div className="text-[#6b6b60]">minimize MSE via ∇GD</div>
        </div>
      </div>
    </div>
  );
}

function LinearRegressionVizWrapper({ s }: { s: LRState }) {
  const done = s.state.converged || s.state.iteration >= 2000;
  return (
    <>
      <LinearRegressionViz s={s} />
      {s.state.iteration === 0 && (
        <p className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] tracking-widest uppercase text-[#6b6b60]/50 pointer-events-none select-none whitespace-nowrap">
          Press Step or Run to fit the line
        </p>
      )}
      {done && (
        <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-[#ddddd4] shadow-sm text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: C.green }} />
          <span className="font-medium" style={{ color: C.green }}>Fitted</span>
        </div>
      )}
      <div className="absolute bottom-3 right-4 flex gap-4 text-[10px] text-[#6b6b60] pointer-events-none">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: C.blue }} />data</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-0.5 rounded" style={{ background: C.orange }} />fit line</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-px border-t border-dashed" style={{ borderColor: C.orange, opacity: 0.5 }} />residual</span>
      </div>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// NEURAL NETWORK — shared state hook
// ─────────────────────────────────────────────────────────────────────────────
type NNPhase = "idle" | "inputs" | "hidden" | "outputs" | "done";

function useNeuralNetState() {
  const [weights, setWeights]           = useState<NNWeights>(() => randomWeights());
  const [inputs, setInputs]             = useState<[number, number]>([0.6, 0.4]);
  const [activationFn, setActivationFn] = useState<ActivationFn>("relu");
  const [activations, setActivations]   = useState<NNActivations | null>(null);
  const [phase, setPhase]               = useState<NNPhase>("idle");

  const timerRef      = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Keep refs so callbacks always read fresh values
  const weightsRef    = useRef(weights);
  const inputsRef     = useRef(inputs);
  const activationRef = useRef(activationFn);
  weightsRef.current    = weights;
  inputsRef.current     = inputs;
  activationRef.current = activationFn;

  const runForwardPass = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    const acts = forwardPass(inputsRef.current, weightsRef.current, activationRef.current);
    setActivations(acts);
    setPhase("inputs");
    timerRef.current = setTimeout(() => {
      setPhase("hidden");
      timerRef.current = setTimeout(() => {
        setPhase("outputs");
        timerRef.current = setTimeout(() => setPhase("done"), 600);
      }, 700);
    }, 500);
  }, []);

  const handleForwardPass = useCallback(() => {
    if (phase !== "idle" && phase !== "done") return;
    runForwardPass();
  }, [phase, runForwardPass]);

  const handleNewWeights = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    const w = randomWeights();
    setWeights(w);
    // Immediately recompute with new weights
    const acts = forwardPass(inputsRef.current, w, activationRef.current);
    setActivations(acts);
    setPhase("done");
  }, []);

  const handleReset = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setActivations(null);
    setPhase("idle");
  }, []);

  // Recompute on input or activation-fn change (only when already showing results)
  useEffect(() => {
    setActivations(forwardPass(inputsRef.current, weightsRef.current, activationRef.current));
    // If we're mid-animation, jump to done
    setPhase((prev) => (prev !== "idle" ? "done" : "idle"));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputs, activationFn]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  return {
    weights, inputs, setInputs, activationFn, setActivationFn,
    activations, phase, handleForwardPass, handleNewWeights, handleReset,
  };
}

type NNState = ReturnType<typeof useNeuralNetState>;

function NeuralNetViz({ s }: { s: NNState }) {
  const W = 480, H = 320;
  const { weights, activations, phase, activationFn } = s;

  // Layer x positions: input, hidden, output
  const layerX = [72, 220, 360];
  const nodeY = (layer: number, idx: number, count: number) => {
    const spacing = Math.min(90, (H - 60) / count);
    const total = spacing * (count - 1);
    return H / 2 - total / 2 + idx * spacing;
  };

  function nodeColor(layer: "input" | "hidden" | "output", idx: number): string {
    const active =
      (layer === "input"  && (phase === "inputs"  || phase === "hidden" || phase === "outputs" || phase === "done")) ||
      (layer === "hidden" && (phase === "hidden"  || phase === "outputs" || phase === "done")) ||
      (layer === "output" && (phase === "outputs" || phase === "done"));
    if (!active || !activations) return "#eeeee8";
    let val = 0;
    if (layer === "input")  val = activations.inputs[idx];
    else if (layer === "hidden") val = activations.hidden[idx];
    else val = activations.outputs[idx];
    const intensity = Math.min(1, Math.max(0, val));
    return `rgba(59, 144, 204, ${0.15 + intensity * 0.85})`;
  }

  function nodeTextColor(layer: "input" | "hidden" | "output", idx: number): string {
    const active =
      (layer === "input"  && (phase === "inputs"  || phase === "hidden" || phase === "outputs" || phase === "done")) ||
      (layer === "hidden" && (phase === "hidden"  || phase === "outputs" || phase === "done")) ||
      (layer === "output" && (phase === "outputs" || phase === "done"));
    if (!active || !activations) return "#aaa";
    let val = 0;
    if (layer === "input")  val = activations.inputs[idx];
    else if (layer === "hidden") val = activations.hidden[idx];
    else val = activations.outputs[idx];
    return val > 0.5 ? "white" : "#1a1a1a";
  }

  function activationVal(layer: "input" | "hidden" | "output", idx: number): string {
    if (!activations) return "";
    const active =
      (layer === "input"  && (phase === "inputs"  || phase === "hidden" || phase === "outputs" || phase === "done")) ||
      (layer === "hidden" && (phase === "hidden"  || phase === "outputs" || phase === "done")) ||
      (layer === "output" && (phase === "outputs" || phase === "done"));
    if (!active) return "";
    if (layer === "input")  return activations.inputs[idx].toFixed(2);
    if (layer === "hidden") return activations.hidden[idx].toFixed(2);
    return activations.outputs[idx].toFixed(2);
  }

  const edgeStyle = (w: number) => ({
    opacity: Math.abs(w) * 0.6 + 0.1,
    strokeWidth: Math.abs(w) * 2 + 0.5,
    stroke: w >= 0 ? C.blue : C.orange,
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" style={{ maxHeight: "100%" }}>
      {/* Layer labels */}
      <text x={layerX[0]} y={20} textAnchor="middle" fontSize="9" fill="#6b6b60">Input</text>
      <text x={layerX[1]} y={20} textAnchor="middle" fontSize="9" fill="#6b6b60">Hidden</text>
      <text x={layerX[2]} y={20} textAnchor="middle" fontSize="9" fill="#6b6b60">Output</text>

      {/* Edges: input → hidden */}
      {[0, 1].map((i) => [0, 1, 2].map((j) => {
        const x1 = layerX[0], y1 = nodeY(0, i, 2);
        const x2 = layerX[1], y2 = nodeY(1, j, 3);
        const s = edgeStyle(weights.w1[j][i]);
        const active = phase === "hidden" || phase === "outputs" || phase === "done";
        return (
          <motion.line key={`i${i}h${j}`}
            x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={s.stroke} strokeWidth={s.strokeWidth}
            animate={{ opacity: active ? s.opacity : 0.06 }}
            transition={{ duration: 0.35 }}
          />
        );
      }))}

      {/* Edges: hidden → output */}
      {[0, 1, 2].map((i) => [0, 1].map((j) => {
        const x1 = layerX[1], y1 = nodeY(1, i, 3);
        const x2 = layerX[2], y2 = nodeY(2, j, 2);
        const s = edgeStyle(weights.w2[j][i]);
        const active = phase === "outputs" || phase === "done";
        return (
          <motion.line key={`h${i}o${j}`}
            x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={s.stroke} strokeWidth={s.strokeWidth}
            animate={{ opacity: active ? s.opacity : 0.06 }}
            transition={{ duration: 0.35 }}
          />
        );
      }))}

      {/* Input nodes */}
      {[0, 1].map((i) => {
        const cx = layerX[0];
        const cy = nodeY(0, i, 2);
        const active = phase === "inputs" || phase === "hidden" || phase === "outputs" || phase === "done";
        return (
          <motion.g key={`in${i}`}>
            <motion.circle cx={cx} cy={cy} r={22}
              fill={nodeColor("input", i)}
              stroke={active ? C.blue : "#ddddd4"}
              strokeWidth={active ? 2 : 1}
              animate={{ fill: nodeColor("input", i) }}
              transition={{ duration: 0.4 }}
            />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize="10"
              fill={nodeTextColor("input", i)} fontFamily="monospace" fontWeight="600">
              {activationVal("input", i) || `x${i + 1}`}
            </text>
          </motion.g>
        );
      })}

      {/* Hidden nodes */}
      {[0, 1, 2].map((i) => {
        const cx = layerX[1];
        const cy = nodeY(1, i, 3);
        const active = phase === "hidden" || phase === "outputs" || phase === "done";
        return (
          <motion.g key={`hid${i}`}>
            <motion.circle cx={cx} cy={cy} r={22}
              fill={nodeColor("hidden", i)}
              stroke={active ? C.purple : "#ddddd4"}
              strokeWidth={active ? 2 : 1}
              animate={{ fill: nodeColor("hidden", i) }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
            />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize="10"
              fill={nodeTextColor("hidden", i)} fontFamily="monospace" fontWeight="600">
              {activationVal("hidden", i) || `h${i + 1}`}
            </text>
          </motion.g>
        );
      })}

      {/* Output nodes */}
      {[0, 1].map((i) => {
        const cx = layerX[2];
        const cy = nodeY(2, i, 2);
        const active = phase === "outputs" || phase === "done";
        return (
          <motion.g key={`out${i}`}>
            <motion.circle cx={cx} cy={cy} r={22}
              fill={nodeColor("output", i)}
              stroke={active ? C.green : "#ddddd4"}
              strokeWidth={active ? 2 : 1}
              animate={{ fill: nodeColor("output", i) }}
              transition={{ duration: 0.4 }}
            />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize="10"
              fill={nodeTextColor("output", i)} fontFamily="monospace" fontWeight="600">
              {activationVal("output", i) || `y${i + 1}`}
            </text>
          </motion.g>
        );
      })}

      {/* Activation fn label */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="9" fill="#6b6b60">
        {activationFn === "relu" ? "ReLU hidden" : "Sigmoid hidden"} · Sigmoid output
      </text>

      {/* Weight legend */}
      <g transform={`translate(${W - 79}, ${H - 52})`}>
        <rect x={-4} y={-4} width={79} height={46} rx={4} fill="white" opacity={0.85} />
        <line x1={0} y1={8} x2={20} y2={8} stroke={C.blue} strokeWidth="2" />
        <text x={24} y={12} fontSize="8" fill="#6b6b60">positive w</text>
        <line x1={0} y1={24} x2={20} y2={24} stroke={C.orange} strokeWidth="2" />
        <text x={24} y={28} fontSize="8" fill="#6b6b60">negative w</text>
        <text x={0} y={42} fontSize="7" fill="#aaa">thickness = |w|</text>
      </g>
    </svg>
  );
}

function NeuralNetworkSidebar({ s }: { s: NNState }) {
  return (
    <div className="p-4 sm:p-6 space-y-6">
      <div>
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Neural Network</h2>
        <div className="w-10 h-px mb-3" style={{ background: C.primary }} />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          A feedforward network with 2 inputs → 3 hidden → 2 outputs.
          Watch activations flow layer by layer during a forward pass.
        </p>
      </div>

      {/* Activations readout */}
      {s.activations && (
        <div>
          <SectionLabel>Activations</SectionLabel>
          <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#6b6b60]">Inputs</span>
              <span className="font-mono" style={{ color: C.blue }}>[{s.activations.inputs.map((v) => v.toFixed(2)).join(", ")}]</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-[#6b6b60]">Hidden</span>
              <span className="font-mono" style={{ color: C.purple }}>[{s.activations.hidden.map((v) => v.toFixed(2)).join(", ")}]</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-[#6b6b60]">Output</span>
              <span className="font-mono" style={{ color: C.green }}>[{s.activations.outputs.map((v) => v.toFixed(2)).join(", ")}]</span>
            </div>
          </div>
        </div>
      )}

      {/* Input sliders */}
      <div>
        <SectionLabel>Inputs</SectionLabel>
        <div className="space-y-3">
          {[0, 1].map((i) => (
            <div key={i}>
              <div className="flex justify-between mb-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">x{i + 1}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: C.blue }}>{s.inputs[i].toFixed(2)}</span>
              </div>
              <input type="range" min={0} max={100} value={Math.round(s.inputs[i] * 100)}
                onChange={(e) => {
                  const newInputs: [number, number] = [...s.inputs] as [number, number];
                  newInputs[i] = Number(e.target.value) / 100;
                  s.setInputs(newInputs);
                }}
                className="w-full" />
            </div>
          ))}
        </div>
      </div>

      {/* Activation function */}
      <div>
        <SectionLabel>Hidden Activation</SectionLabel>
        <div className="grid grid-cols-2 gap-2">
          {(["relu", "sigmoid"] as ActivationFn[]).map((fn) => (
            <motion.button key={fn} whileTap={{ scale: 0.95 }}
              onClick={() => s.setActivationFn(fn)}
              className={`py-2.5 rounded-xl text-xs font-semibold border transition-all duration-200 ${
                s.activationFn === fn
                  ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248]"
                  : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25"
              }`}>
              {fn === "relu" ? "ReLU" : "Sigmoid"}
            </motion.button>
          ))}
        </div>
        <div className="mt-2 rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-2 font-mono text-[10px] text-center text-[#6b6b60]">
          {s.activationFn === "relu" ? "f(x) = max(0, x)" : "f(x) = 1 / (1 + e⁻ˣ)"}
        </div>
      </div>

      {/* Buttons */}
      <div className="flex gap-2 flex-wrap">
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleForwardPass}
          disabled={s.phase !== "idle" && s.phase !== "done"}
          className={`flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200 border ${
            s.phase !== "idle" && s.phase !== "done"
              ? "opacity-35 cursor-not-allowed bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60]"
              : "bg-[#7a6248]/10 border-[#7a6248]/30 text-[#7a6248] hover:bg-[#7a6248]/18 shadow-sm"
          }`}>
          Forward Pass
        </motion.button>
        <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleReset}
          className="flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200 border bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]">
          Reset
        </motion.button>
      </div>
      <motion.button whileTap={{ scale: 0.95 }} onClick={s.handleNewWeights}
        className="w-full py-2.5 rounded-xl text-sm font-semibold border transition-all duration-200 bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]">
        New Weights
      </motion.button>
    </div>
  );
}

function NeuralNetworkVizWrapper({ s }: { s: NNState }) {
  return (
    <>
      <NeuralNetViz s={s} />
      {s.phase === "idle" && (
        <p className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] tracking-widest uppercase text-[#6b6b60]/50 pointer-events-none select-none whitespace-nowrap">
          Press Forward Pass to animate
        </p>
      )}
      {s.phase === "done" && (
        <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-[#ddddd4] shadow-sm text-xs">
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: C.green }} />
          <span className="font-medium" style={{ color: C.green }}>Pass complete</span>
        </div>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROOT COMPONENT — state is owned here, passed to viz + sidebar as props
// ─────────────────────────────────────────────────────────────────────────────
export default function MachineLearningVisualizer() {
  const [topic, setTopic] = useState<Topic>("gradient-descent");

  // Each topic owns its state here so both viz and sidebar share the same instance
  const gdState  = useGradientDescentState();
  const kmState  = useKMeansState();
  const lrState  = useLinRegState();
  const nnState  = useNeuralNetState();

  // Stop any running animation when switching topics
  const prevTopicRef = useRef(topic);
  useEffect(() => {
    if (prevTopicRef.current !== topic) {
      // Stop gradient descent
      gdState.handleReset();
      // Stop k-means (via new-data would be too destructive; just stop running)
      kmState.handleNewData();
      // Stop linear regression
      lrState.handleReset();
      // Stop neural network
      nnState.handleReset();
      prevTopicRef.current = topic;
    }
  // We only want to run this when topic changes, not when handlers change
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topic]);

  return (
    <div
      className="flex flex-col bg-[#f5f5f0] text-[#1a1a1a] overflow-hidden"
      style={{ height: "calc(100vh - 3.5rem)" }}
    >
      <div className="flex-1 flex flex-col lg:flex-row min-h-0">

        {/* ── Visualization main ───────────────────────────────────────────── */}
        <main
          className="order-first lg:order-2 flex-shrink-0 lg:flex-shrink lg:flex-1
                     h-[45vh] lg:h-auto border-b lg:border-b-0 border-[#ddddd4]
                     dot-pattern bg-[#f5f5f0] p-4 sm:p-6 flex flex-col min-h-0 relative overflow-hidden"
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={topic + "-viz"}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="flex-1 flex flex-col min-h-0 relative"
            >
              {topic === "gradient-descent"  && (gdState.view3d ? <GradientDescentViz3D s={gdState} /> : <GradientDescentViz s={gdState} />)}
              {topic === "kmeans"            && <KMeansViz s={kmState} />}
              {topic === "linear-regression" && <LinearRegressionVizWrapper s={lrState} />}
              {topic === "neural-network"    && <NeuralNetworkVizWrapper s={nnState} />}
            </motion.div>
          </AnimatePresence>
        </main>

        {/* ── Sidebar ──────────────────────────────────────────────────────── */}
        <motion.aside
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, delay: 0.1, ease: "easeOut" }}
          className="order-2 lg:order-1 flex-1 lg:flex-none lg:w-80 xl:w-96
                     overflow-y-auto lg:border-r border-[#ddddd4] bg-white"
        >
          {/* Topic picker */}
          <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">Topic</p>
            <div className="grid grid-cols-2 gap-2">
              {TOPICS.map((t, i) => (
                <motion.button
                  key={t.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05, duration: 0.25 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setTopic(t.id)}
                  className={`px-2 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 border text-left leading-tight ${
                    topic === t.id
                      ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248] shadow-sm"
                      : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]"
                  }`}
                >
                  {t.label}
                </motion.button>
              ))}
            </div>
          </div>

          {/* Topic-specific sidebar content */}
          <AnimatePresence mode="wait">
            <motion.div
              key={topic + "-sidebar"}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              {topic === "gradient-descent"  && <GradientDescentSidebar s={gdState} />}
              {topic === "kmeans"            && <KMeansSidebar s={kmState} />}
              {topic === "linear-regression" && <LinearRegressionSidebar s={lrState} />}
              {topic === "neural-network"    && <NeuralNetworkSidebar s={nnState} />}
            </motion.div>
          </AnimatePresence>
        </motion.aside>
      </div>
    </div>
  );
}
