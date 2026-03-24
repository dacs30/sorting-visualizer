"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Vec2, Mat2,
  vecAdd, vecLength, vecDot, vecProject, vecNormalize, matMulVec,
  matDeterminant, matLerp, MAT_IDENTITY,
  eigen2x2, classifyTransform,
  MATRIX_PRESETS, EIGEN_PRESETS,
  toSVG, fmt, DEG, RAD,
} from "../lib/linearAlgebra";

// ── Palette ───────────────────────────────────────────────────────────────────

const C = {
  primary:  "#7a6248",
  blue:     "#3b90cc",
  orange:   "#c86030",
  green:    "#3a9a50",
  purple:   "#a090c8",
  gold:     "#b89020",
} as const;

// ── Topic list ────────────────────────────────────────────────────────────────

type Topic = "vectors" | "transform" | "dot" | "eigen";

const TOPICS: { id: Topic; label: string; short: string }[] = [
  { id: "vectors",   label: "Vector Addition",      short: "Vectors" },
  { id: "transform", label: "Linear Transformation", short: "Transform" },
  { id: "dot",       label: "Dot Product",           short: "Dot ·" },
  { id: "eigen",     label: "Eigenvectors",          short: "Eigen" },
];

// ── Shared SVG helpers ────────────────────────────────────────────────────────

/** Arrow marker defs */
function Defs() {
  return (
    <defs>
      {(["blue", "purple", "green", "gold", "orange", "primary"] as const).map((k) => (
        <marker
          key={k}
          id={`arrow-${k}`}
          markerWidth="10" markerHeight="10"
          refX="9" refY="5"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path d="M0,1 L0,9 L9,5 z" fill={C[k]} />
        </marker>
      ))}
      <marker id="arrow-muted" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse">
        <path d="M0,1 L0,9 L9,5 z" fill="#6b6b60" />
      </marker>
    </defs>
  );
}

interface GridProps {
  cx: number; cy: number; scale: number;
  w: number; h: number;
  range?: number;
}

function CoordGrid({ cx, cy, scale, w, h, range = 4 }: GridProps) {
  const lines: React.ReactNode[] = [];
  for (let i = -range; i <= range; i++) {
    const x = cx + i * scale;
    const y = cy - i * scale;
    const isAxis = i === 0;
    lines.push(
      <line key={`v${i}`} x1={x} y1={0} x2={x} y2={h}
        stroke={isAxis ? "#9a8878" : "#ddddd4"} strokeWidth={isAxis ? 1.5 : 0.75} />,
      <line key={`h${i}`} x1={0} y1={y} x2={w} y2={y}
        stroke={isAxis ? "#9a8878" : "#ddddd4"} strokeWidth={isAxis ? 1.5 : 0.75} />,
    );
    if (i !== 0 && Math.abs(i) <= range) {
      lines.push(
        <text key={`xl${i}`} x={x} y={cy + 14} textAnchor="middle"
          fontSize="9" fill="#9a8878" fontFamily="monospace">{i}</text>,
        <text key={`yl${i}`} x={cx - 8} y={y + 3} textAnchor="end"
          fontSize="9" fill="#9a8878" fontFamily="monospace">{i}</text>,
      );
    }
  }
  return <g>{lines}</g>;
}

interface ArrowProps {
  from: Vec2; to: Vec2;
  color: string;
  colorKey: keyof typeof C | "muted";
  dashed?: boolean;
  opacity?: number;
  strokeWidth?: number;
}

function Arrow({ from, to, color, colorKey, dashed, opacity = 1, strokeWidth = 2 }: ArrowProps) {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 4) return null;
  // Shorten end by arrowhead size so arrowhead sits at tip
  const ux = dx / len, uy = dy / len;
  const shortenBy = 9;
  const end: Vec2 = [to[0] - ux * shortenBy, to[1] - uy * shortenBy];
  return (
    <line
      x1={from[0]} y1={from[1]}
      x2={end[0]} y2={end[1]}
      stroke={color}
      strokeWidth={strokeWidth}
      strokeDasharray={dashed ? "5 4" : undefined}
      opacity={opacity}
      markerEnd={`url(#arrow-${colorKey})`}
    />
  );
}

// ── Draggable vector tip ──────────────────────────────────────────────────────

interface DragTipProps {
  pos: Vec2;            // SVG pixel position
  color: string;
  onDrag: (pos: Vec2) => void;
}

function DragTip({ pos, color, onDrag }: DragTipProps) {
  const dragging = useRef(false);
  const circleRef = useRef<SVGCircleElement>(null);
  // Keep a ref to onDrag so touch handlers (registered once) always see latest value
  const onDragRef = useRef(onDrag);
  useEffect(() => { onDragRef.current = onDrag; }, [onDrag]);

  // Register touch listeners via useEffect so we can pass { passive: false }
  useEffect(() => {
    const el = circleRef.current;
    if (!el) return;

    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      dragging.current = true;
      const svg = el.ownerSVGElement as SVGSVGElement;

      const onMove = (ev: TouchEvent) => {
        if (!dragging.current || ev.touches.length === 0) return;
        ev.preventDefault();
        const rect = svg.getBoundingClientRect();
        onDragRef.current([ev.touches[0].clientX - rect.left, ev.touches[0].clientY - rect.top]);
      };
      const onUp = () => {
        dragging.current = false;
        window.removeEventListener("touchmove", onMove);
        window.removeEventListener("touchend", onUp);
      };
      window.addEventListener("touchmove", onMove, { passive: false });
      window.addEventListener("touchend", onUp);
    };

    el.addEventListener("touchstart", onTouchStart, { passive: false });
    return () => {
      el.removeEventListener("touchstart", onTouchStart);
    };
  }, []);

  const handleMouseDown = (e: React.MouseEvent<SVGCircleElement>) => {
    e.preventDefault();
    dragging.current = true;
    const svg = (e.currentTarget.ownerSVGElement) as SVGSVGElement;
    const onMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const rect = svg.getBoundingClientRect();
      onDragRef.current([ev.clientX - rect.left, ev.clientY - rect.top]);
    };
    const onUp = () => {
      dragging.current = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <circle
      ref={circleRef}
      cx={pos[0]} cy={pos[1]} r={8}
      fill={color} fillOpacity={0.2}
      stroke={color} strokeWidth={2}
      style={{ cursor: "grab" }}
      onMouseDown={handleMouseDown}
    />
  );
}

// ── Shared hooks ──────────────────────────────────────────────────────────────

function useDims(svgRef: React.RefObject<SVGSVGElement | null>) {
  const [dims, setDims] = useState({ w: 500, h: 400 });
  useEffect(() => {
    const el = svgRef.current?.parentElement ?? svgRef.current;
    if (!el) return;
    // Read actual size immediately on mount — don't wait for first resize event
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) setDims({ w: r.width, h: r.height });
    const obs = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setDims({ w: width, h: height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, [svgRef]);
  return dims;
}

// ── Shared state types ────────────────────────────────────────────────────────

interface VectorsState {
  v1: Vec2;
  v2: Vec2;
  setV1: (v: Vec2) => void;
  setV2: (v: Vec2) => void;
}

interface MatInputs {
  a: string; b: string; c: string; d: string;
}

interface TransformState {
  matInputs: MatInputs;
  setMatInputs: (inputs: MatInputs) => void;
  targetMatrix: Mat2;
  t: number;
  animating: boolean;
  applied: boolean;
  applyMatrix: (overrideMatrix?: Mat2) => void;
  resetTransform: () => void;
}

interface DotState {
  angle1: number;
  angle2: number;
  setAngle1: (a: number) => void;
  setAngle2: (a: number) => void;
}

type EigenPhase = "before" | "after" | "highlight";

interface EigenState {
  presetKey: string;
  setPresetKey: (k: string) => void;
  phase: EigenPhase;
  animT: number;
  goToPhase: (p: EigenPhase) => void;
}

// ── 1. Vector Addition ────────────────────────────────────────────────────────

interface VizVectorAdditionProps {
  state: VectorsState;
}

function VizVectorAddition({ state }: VizVectorAdditionProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dims = useDims(svgRef);

  const { v1, v2, setV1, setV2 } = state;

  const scale = Math.min(dims.w, dims.h) / 9;
  const cx = dims.w / 2;
  const cy = dims.h / 2;

  const result = vecAdd(v1, v2);

  const mathToSVG = useCallback((v: Vec2): Vec2 => toSVG(v, cx, cy, scale), [cx, cy, scale]);
  const svgToMath = useCallback((p: Vec2): Vec2 => [(p[0] - cx) / scale, -(p[1] - cy) / scale], [cx, cy, scale]);

  const clampVec = (v: Vec2): Vec2 => {
    const len = vecLength(v);
    const maxLen = 3.8;
    if (len > maxLen) { const n = vecNormalize(v); return [n[0] * maxLen, n[1] * maxLen]; }
    return v;
  };

  const tip1 = mathToSVG(v1);
  const tip2 = mathToSVG(v2);
  const tipR = mathToSVG(result);
  // parallelogram dashed lines: v1 tip → result, v2 tip → result
  const pA = mathToSVG(v1);
  const pC = mathToSVG(v2);
  const origin: Vec2 = [cx, cy];

  return (
    <svg ref={svgRef} width="100%" height="100%" className="w-full h-full">
      <Defs />
      <CoordGrid cx={cx} cy={cy} scale={scale} w={dims.w} h={dims.h} range={4} />

      {/* Parallelogram */}
      <line x1={pA[0]} y1={pA[1]} x2={tipR[0]} y2={tipR[1]}
        stroke={C.purple} strokeWidth={1} strokeDasharray="5 4" opacity={0.45} />
      <line x1={pC[0]} y1={pC[1]} x2={tipR[0]} y2={tipR[1]}
        stroke={C.blue} strokeWidth={1} strokeDasharray="5 4" opacity={0.45} />

      <Arrow from={origin} to={tip1} color={C.blue} colorKey="blue" />
      <Arrow from={origin} to={tip2} color={C.purple} colorKey="purple" />
      <Arrow from={origin} to={tipR} color={C.green} colorKey="green" strokeWidth={2.5} />

      <text x={tip1[0] + 8} y={tip1[1] - 8} fontSize="12" fill={C.blue} fontWeight="600">v₁</text>
      <text x={tip2[0] + 8} y={tip2[1] - 8} fontSize="12" fill={C.purple} fontWeight="600">v₂</text>
      <text x={tipR[0] + 8} y={tipR[1] - 8} fontSize="12" fill={C.green} fontWeight="600">v₁+v₂</text>

      <DragTip pos={tip1} color={C.blue}
        onDrag={(p) => setV1(clampVec(svgToMath(p)))} />
      <DragTip pos={tip2} color={C.purple}
        onDrag={(p) => setV2(clampVec(svgToMath(p)))} />

      {/* Legend */}
      <text x={12} y={dims.h - 14} fontSize="10" fill="#9a8878">Drag the dots to move vectors</text>
    </svg>
  );
}

// ── 2. Linear Transformation ──────────────────────────────────────────────────

interface VizTransformProps {
  state: TransformState;
}

function VizTransform({ state }: VizTransformProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dims = useDims(svgRef);

  const { matInputs, setMatInputs, targetMatrix, t, animating, applied, applyMatrix, resetTransform } = state;

  const scale = Math.min(dims.w, dims.h) / 9;
  const cx = dims.w / 2;
  const cy = dims.h / 2;

  const currentMat = matLerp(MAT_IDENTITY, targetMatrix, t);
  const mathToSVG = (v: Vec2): Vec2 => toSVG(v, cx, cy, scale);
  const applyAndSVG = (v: Vec2): Vec2 => mathToSVG(matMulVec(currentMat, v));
  const origin: Vec2 = mathToSVG([0, 0]);

  const range = 3;
  const gridLines: Array<[Vec2, Vec2]> = [];
  for (let i = -range; i <= range; i++) {
    gridLines.push([[-range, i], [range, i]]);
    gridLines.push([[i, -range], [i, range]]);
  }

  return (
    <div className="flex flex-col w-full h-full">
      <div className="flex-shrink-0 flex items-center justify-center gap-2 p-3 flex-wrap">
        <div className="flex items-center gap-1">
          <span className="text-[#9a8878] font-mono text-sm">[</span>
          <div className="grid grid-cols-2 gap-1">
            {(["a","b","c","d"] as const).map((k) => (
              <input key={k} type="number" step="0.1" value={matInputs[k]}
                onChange={(e) => setMatInputs({ ...matInputs, [k]: e.target.value })}
                className="w-12 text-center text-xs font-mono rounded-lg border border-[#ddddd4]
                           bg-[#f5f5f0] text-[#1a1a1a] py-1 px-1 focus:outline-none focus:border-[#7a6248]"
              />
            ))}
          </div>
          <span className="text-[#9a8878] font-mono text-sm">]</span>
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {Object.entries(MATRIX_PRESETS).slice(1).map(([key, { label, matrix }]) => (
            <motion.button key={key} whileTap={{ scale: 0.95 }}
              onClick={() => {
                setMatInputs({
                  a: String(matrix[0][0]), b: String(matrix[0][1]),
                  c: String(matrix[1][0]), d: String(matrix[1][1]),
                });
                applyMatrix(matrix);
              }}
              className="px-2 py-1 rounded-lg text-[10px] font-medium border
                         bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25"
            >
              {label}
            </motion.button>
          ))}
        </div>
        <div className="flex gap-1.5">
          <motion.button whileTap={{ scale: 0.95 }} onClick={() => applyMatrix()} disabled={animating}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold border
                       bg-[#7a6248]/10 border-[#7a6248]/30 text-[#7a6248]
                       hover:bg-[#7a6248]/18 disabled:opacity-40">
            Apply
          </motion.button>
          <motion.button whileTap={{ scale: 0.95 }} onClick={resetTransform}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold border
                       bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25">
            Reset
          </motion.button>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <svg ref={svgRef} width="100%" height="100%" className="w-full h-full">
          <Defs />
          <line x1={0} y1={cy} x2={dims.w} y2={cy} stroke="#9a8878" strokeWidth={1} />
          <line x1={cx} y1={0} x2={cx} y2={dims.h} stroke="#9a8878" strokeWidth={1} />

          {gridLines.map(([start, end], i) => {
            const s = applyAndSVG(start);
            const e = applyAndSVG(end);
            return <line key={i} x1={s[0]} y1={s[1]} x2={e[0]} y2={e[1]}
              stroke={t < 0.01 ? "#ddddd4" : "#c8b8a8"} strokeWidth={0.75} opacity={0.7} />;
          })}

          <Arrow from={origin} to={applyAndSVG([1, 0])} color={C.blue} colorKey="blue" strokeWidth={2.5} />
          <Arrow from={origin} to={applyAndSVG([0, 1])} color={C.orange} colorKey="orange" strokeWidth={2.5} />

          <text x={applyAndSVG([1, 0])[0] + 8} y={applyAndSVG([1, 0])[1] - 6} fontSize="12" fill={C.blue} fontWeight="600">î</text>
          <text x={applyAndSVG([0, 1])[0] + 8} y={applyAndSVG([0, 1])[1] - 6} fontSize="12" fill={C.orange} fontWeight="600">ĵ</text>

          {applied && (
            <text x={12} y={dims.h - 12} fontSize="11" fill="#9a8878" fontFamily="monospace">
              {`det = ${fmt(matDeterminant(targetMatrix))}  ·  ${classifyTransform(targetMatrix)}`}
            </text>
          )}
        </svg>
      </div>
    </div>
  );
}

// ── 3. Dot Product ────────────────────────────────────────────────────────────

interface VizDotProps {
  state: DotState;
}

function VizDot({ state }: VizDotProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dims = useDims(svgRef);

  const { angle1, angle2 } = state;

  const scale = Math.min(dims.w, dims.h) / 9;
  const cx = dims.w / 2;
  const cy = dims.h / 2;
  const vLen = 2.5;

  const v1: Vec2 = [vLen * Math.cos(angle1 * DEG), vLen * Math.sin(angle1 * DEG)];
  const v2: Vec2 = [vLen * Math.cos(angle2 * DEG), vLen * Math.sin(angle2 * DEG)];
  const dot = vecDot(v1, v2);
  const angleBetween = Math.acos(Math.max(-1, Math.min(1, dot / (vLen * vLen))));
  const proj = vecProject(v1, v2);

  const mathToSVG = (v: Vec2): Vec2 => toSVG(v, cx, cy, scale);
  const origin: Vec2 = [cx, cy];
  const tip1 = mathToSVG(v1);
  const tip2 = mathToSVG(v2);
  const projTip = mathToSVG(proj);
  const perp1 = mathToSVG(v1);
  const perp2 = mathToSVG(proj);

  const arcR = 40;
  const arcStart = { x: cx + arcR * Math.cos(-angle1 * DEG), y: cy + arcR * Math.sin(-angle1 * DEG) };
  const arcEnd   = { x: cx + arcR * Math.cos(-angle2 * DEG), y: cy + arcR * Math.sin(-angle2 * DEG) };
  const diff = ((angle2 - angle1 + 360) % 360);
  const largeArc = diff > 180 ? 1 : 0;
  const sweep = diff <= 180 ? 1 : 0;
  const arcPath = `M ${arcStart.x} ${arcStart.y} A ${arcR} ${arcR} 0 ${largeArc} ${sweep} ${arcEnd.x} ${arcEnd.y}`;
  const midAngle = angle1 + diff / 2;

  return (
    <svg ref={svgRef} width="100%" height="100%" className="w-full h-full">
      <Defs />
      <CoordGrid cx={cx} cy={cy} scale={scale} w={dims.w} h={dims.h} range={3} />

      <path d={arcPath} fill="none" stroke="#9a8878" strokeWidth={1} opacity={0.6} />
      <text
        x={cx + (arcR + 14) * Math.cos(-midAngle * DEG)}
        y={cy + (arcR + 14) * Math.sin(-midAngle * DEG)}
        fontSize="11" fill="#6b6b60" textAnchor="middle"
      >θ</text>

      {/* Perpendicular foot */}
      <line x1={perp1[0]} y1={perp1[1]} x2={perp2[0]} y2={perp2[1]}
        stroke={C.gold} strokeWidth={1.5} strokeDasharray="4 3" opacity={0.8} />

      <Arrow from={origin} to={projTip} color={C.gold} colorKey="gold" strokeWidth={2} dashed />
      <Arrow from={origin} to={tip1} color={C.blue} colorKey="blue" strokeWidth={2.5} />
      <Arrow from={origin} to={tip2} color={C.purple} colorKey="purple" strokeWidth={2.5} />

      <text x={tip1[0] + 8} y={tip1[1] - 8} fontSize="12" fill={C.blue} fontWeight="600">v₁</text>
      <text x={tip2[0] + 8} y={tip2[1] - 8} fontSize="12" fill={C.purple} fontWeight="600">v₂</text>
      <text x={projTip[0] + 8} y={projTip[1] + 16} fontSize="11" fill={C.gold} fontWeight="600">proj</text>

      <text x={12} y={dims.h - 26} fontSize="11" fill="#6b6b60" fontFamily="monospace">
        {`v₁·v₂ = ${fmt(dot)}   θ = ${fmt(angleBetween * RAD, 1)}°`}
      </text>
      <text x={12} y={dims.h - 10} fontSize="10" fill="#9a8878" fontFamily="monospace">
        {`|v₁||v₂|cos(θ) = ${fmt(vLen)}·${fmt(vLen)}·${fmt(Math.cos(angleBetween), 3)}`}
      </text>
    </svg>
  );
}

// ── 4. Eigenvectors ───────────────────────────────────────────────────────────

interface VizEigenProps {
  state: EigenState;
}

function VizEigen({ state }: VizEigenProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dims = useDims(svgRef);

  const { presetKey, setPresetKey, phase, animT, goToPhase } = state;

  const preset = EIGEN_PRESETS[presetKey];
  const matrix = preset.matrix;
  const eigenResult = eigen2x2(matrix);

  const scale = Math.min(dims.w, dims.h) / 10;
  const cx = dims.w / 2;
  const cy = dims.h / 2;

  const N = 16;
  const unitVecs: Vec2[] = Array.from({ length: N }, (_, i) => {
    const a = (i / N) * 2 * Math.PI;
    return [Math.cos(a), Math.sin(a)];
  });

  const isEigenvec = (v: Vec2): boolean => {
    const tv = matMulVec(matrix, v);
    const tvLen = vecLength(tv);
    const vLen = vecLength(v);
    if (tvLen < 1e-8 || vLen < 1e-8) return false;
    const tvN: Vec2 = [tv[0] / tvLen, tv[1] / tvLen];
    const vN: Vec2 = [v[0] / vLen, v[1] / vLen];
    const d = Math.abs(tvN[0] * vN[0] + tvN[1] * vN[1]);
    return d > 0.997;
  };

  const mathToSVG = (v: Vec2): Vec2 => toSVG(v, cx, cy, scale);
  const currentMat = matLerp(MAT_IDENTITY, matrix, animT);
  const origin: Vec2 = [cx, cy];
  const eigFlag = unitVecs.map(isEigenvec);

  return (
    <div className="flex flex-col w-full h-full">
      <div className="flex-shrink-0 flex items-center justify-center gap-2 p-3 flex-wrap">
        <div className="flex gap-1.5 flex-wrap">
          {Object.entries(EIGEN_PRESETS).map(([key, { label }]) => (
            <motion.button key={key} whileTap={{ scale: 0.95 }}
              onClick={() => {
                setPresetKey(key);
                goToPhase("before");
              }}
              className={`px-2.5 py-1.5 rounded-xl text-xs font-medium border transition-all
                ${presetKey === key
                  ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248]"
                  : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25"}`}
            >
              {label}
            </motion.button>
          ))}
        </div>
        <div className="flex gap-1.5">
          {(["before", "after", "highlight"] as EigenPhase[]).map((p) => (
            <motion.button key={p} whileTap={{ scale: 0.95 }}
              onClick={() => goToPhase(p)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all capitalize
                ${phase === p
                  ? "bg-[#7a6248]/10 border-[#7a6248]/30 text-[#7a6248]"
                  : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25"}`}
            >
              {p}
            </motion.button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <svg ref={svgRef} width="100%" height="100%" className="w-full h-full">
          <Defs />
          <CoordGrid cx={cx} cy={cy} scale={scale} w={dims.w} h={dims.h} range={4} />

          {unitVecs.map((v, i) => {
            const transformed = matMulVec(currentMat, v);
            const tip = mathToSVG(transformed);
            const isEigen = eigFlag[i];
            const isHigh = phase === "highlight" && isEigen;
            const color = isHigh ? C.gold : (animT < 0.01 ? C.blue : C.purple);
            const colorKey = isHigh ? "gold" : (animT < 0.01 ? "blue" : "purple");
            return (
              <Arrow key={i} from={origin} to={tip}
                color={color} colorKey={colorKey as keyof typeof C}
                strokeWidth={isHigh ? 2.5 : 1.5}
                opacity={isHigh ? 1 : phase === "highlight" ? 0.25 : 0.65}
              />
            );
          })}

          {phase === "highlight" && eigenResult.isReal && (
            <>
              {[eigenResult.vec1, eigenResult.vec2].map((ev, i) => {
                const evTransformed = matMulVec(matrix, ev);
                const evLen = vecLength(evTransformed);
                const scaled: Vec2 = evLen > 0.1 ? [evTransformed[0] / evLen * 3.5, evTransformed[1] / evLen * 3.5] : [ev[0] * 3.5, ev[1] * 3.5];
                const neg: Vec2 = [-scaled[0], -scaled[1]];
                return (
                  <line key={i}
                    x1={mathToSVG(neg)[0]} y1={mathToSVG(neg)[1]}
                    x2={mathToSVG(scaled)[0]} y2={mathToSVG(scaled)[1]}
                    stroke={C.gold} strokeWidth={1} strokeDasharray="5 3" opacity={0.4}
                  />
                );
              })}
            </>
          )}

          <text x={12} y={dims.h - 26} fontSize="11" fill="#9a8878" fontFamily="monospace">
            {eigenResult.isReal
              ? `λ₁=${fmt(eigenResult.lambda1)}  λ₂=${fmt(eigenResult.lambda2)}`
              : "Complex eigenvalues"}
          </text>
          <text x={12} y={dims.h - 10} fontSize="10" fill="#9a8878">
            {phase === "before" ? "Unit circle" : phase === "after" ? "After transform" : "Gold = eigenvectors"}
          </text>
        </svg>
      </div>
    </div>
  );
}

// ── Sidebar sections ──────────────────────────────────────────────────────────

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">
      {children}
    </p>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[#ddddd4] last:border-0">
      <span className="text-xs text-[#6b6b60]">{label}</span>
      <span className="font-mono text-xs font-semibold text-[#3b90cc]">{value}</span>
    </div>
  );
}

// ── Topic sidebars ────────────────────────────────────────────────────────────

interface VectorsSidebarProps {
  state: VectorsState;
}

function VectorsSidebar({ state }: VectorsSidebarProps) {
  const { v1, v2, setV1, setV2 } = state;
  const result = vecAdd(v1, v2);

  return (
    <>
      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Vector Addition</h2>
        <div className="w-10 h-px bg-[#7a6248] mb-3" />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          Two vectors are added tip-to-tail. The <span style={{ color: C.green }}>result</span> is shown
          with the parallelogram law — drag the colored dots to explore.
        </p>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4] space-y-4">
        <SectionHeader>v₁ (blue)</SectionHeader>
        {(["x", "y"] as const).map((axis, ai) => (
          <div key={axis}>
            <div className="flex justify-between mb-1.5">
              <span className="text-xs text-[#6b6b60]">{axis}</span>
              <span className="text-xs font-mono font-semibold text-[#3b90cc]">{fmt(v1[ai])}</span>
            </div>
            <input type="range" min={-3.5} max={3.5} step={0.1}
              value={v1[ai]}
              onChange={(e) => {
                const val = Number(e.target.value);
                setV1(ai === 0 ? [val, v1[1]] : [v1[0], val]);
              }}
              className="w-full"
            />
          </div>
        ))}
        <SectionHeader>v₂ (purple)</SectionHeader>
        {(["x", "y"] as const).map((axis, ai) => (
          <div key={axis}>
            <div className="flex justify-between mb-1.5">
              <span className="text-xs text-[#6b6b60]">{axis}</span>
              <span className="text-xs font-mono font-semibold" style={{ color: C.purple }}>{fmt(v2[ai])}</span>
            </div>
            <input type="range" min={-3.5} max={3.5} step={0.1}
              value={v2[ai]}
              onChange={(e) => {
                const val = Number(e.target.value);
                setV2(ai === 0 ? [val, v2[1]] : [v2[0], val]);
              }}
              className="w-full"
            />
          </div>
        ))}
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <SectionHeader>Result v₁ + v₂</SectionHeader>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] px-4 py-1">
          <InfoRow label="x" value={fmt(result[0])} />
          <InfoRow label="y" value={fmt(result[1])} />
          <InfoRow label="|v₁+v₂|" value={fmt(vecLength(result))} />
        </div>
      </div>

      <div className="p-4 sm:p-6">
        <SectionHeader>Formula</SectionHeader>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 font-mono text-xs text-[#6b6b60] space-y-1">
          <div>v₁ = ({fmt(v1[0])}, {fmt(v1[1])})</div>
          <div>v₂ = ({fmt(v2[0])}, {fmt(v2[1])})</div>
          <div style={{ color: C.green }}>v₁+v₂ = ({fmt(result[0])}, {fmt(result[1])})</div>
        </div>
      </div>
    </>
  );
}

interface TransformSidebarProps {
  state: TransformState;
}

function TransformSidebar({ state }: TransformSidebarProps) {
  const { matInputs, setMatInputs, applyMatrix } = state;

  const a = parseFloat(matInputs.a) || 0;
  const b = parseFloat(matInputs.b) || 0;
  const c = parseFloat(matInputs.c) || 0;
  const d = parseFloat(matInputs.d) || 0;
  const mat: Mat2 = [[a, b], [c, d]];
  const det = matDeterminant(mat);

  return (
    <>
      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Linear Transformation</h2>
        <div className="w-10 h-px bg-[#7a6248] mb-3" />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          Enter a 2×2 matrix and press Apply to watch the grid and basis vectors animate
          to their transformed positions.
        </p>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <SectionHeader>Presets</SectionHeader>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(MATRIX_PRESETS).map(([key, { label, matrix }]) => (
            <motion.button
              key={key}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                setMatInputs({
                  a: String(matrix[0][0]), b: String(matrix[0][1]),
                  c: String(matrix[1][0]), d: String(matrix[1][1]),
                });
                applyMatrix(matrix);
              }}
              className="px-2 py-2 rounded-xl text-xs font-medium border
                         bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60]
                         hover:border-[#7a6248]/25 hover:text-[#1a1a1a] transition-colors"
            >
              {label}
            </motion.button>
          ))}
        </div>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <SectionHeader>Properties</SectionHeader>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] px-4 py-1">
          <InfoRow label="Determinant" value={fmt(det)} />
          <InfoRow label="Trace" value={fmt(a + d)} />
          <InfoRow label="Type" value={classifyTransform(mat)} />
        </div>
      </div>

      <div className="p-4 sm:p-6">
        <SectionHeader>Basis vectors after</SectionHeader>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 font-mono text-xs space-y-1.5">
          <div><span style={{ color: C.blue }}>î</span><span className="text-[#6b6b60]"> → ({fmt(a)}, {fmt(c)})</span></div>
          <div><span style={{ color: C.orange }}>ĵ</span><span className="text-[#6b6b60]"> → ({fmt(b)}, {fmt(d)})</span></div>
        </div>
      </div>
    </>
  );
}

interface DotSidebarProps {
  state: DotState;
}

function DotSidebar({ state }: DotSidebarProps) {
  const { angle1, angle2, setAngle1, setAngle2 } = state;

  const vLen = 2.5;
  const v1: Vec2 = [vLen * Math.cos(angle1 * DEG), vLen * Math.sin(angle1 * DEG)];
  const v2: Vec2 = [vLen * Math.cos(angle2 * DEG), vLen * Math.sin(angle2 * DEG)];
  const dot = vecDot(v1, v2);
  const angleBetween = Math.acos(Math.max(-1, Math.min(1, dot / (vLen * vLen))));
  const projLen = dot / vLen; // scalar projection of v1 onto v2

  return (
    <>
      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Dot Product</h2>
        <div className="w-10 h-px bg-[#7a6248] mb-3" />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          The dot product measures how aligned two vectors are.
          v₁·v₂ = |v₁||v₂|cos(θ). The <span style={{ color: C.gold }}>gold dashed line</span> shows
          the projection of v₁ onto v₂.
        </p>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4] space-y-4">
        <div>
          <div className="flex justify-between mb-1.5">
            <span className="text-xs text-[#6b6b60]">Angle of v₁</span>
            <span className="text-xs font-mono font-semibold" style={{ color: C.blue }}>{angle1}°</span>
          </div>
          <input type="range" min={0} max={360} step={1} value={angle1}
            onChange={(e) => setAngle1(Number(e.target.value))} className="w-full" />
        </div>
        <div>
          <div className="flex justify-between mb-1.5">
            <span className="text-xs text-[#6b6b60]">Angle of v₂</span>
            <span className="text-xs font-mono font-semibold" style={{ color: C.purple }}>{angle2}°</span>
          </div>
          <input type="range" min={0} max={360} step={1} value={angle2}
            onChange={(e) => setAngle2(Number(e.target.value))} className="w-full" />
        </div>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <SectionHeader>Values</SectionHeader>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] px-4 py-1">
          <InfoRow label="v₁ · v₂" value={fmt(dot)} />
          <InfoRow label="θ (between)" value={`${fmt(angleBetween * RAD, 1)}°`} />
          <InfoRow label="Scalar proj" value={fmt(projLen)} />
        </div>
      </div>

      <div className="p-4 sm:p-6">
        <SectionHeader>Formula</SectionHeader>
        <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 font-mono text-xs text-[#6b6b60] leading-relaxed">
          v₁·v₂ = |v₁||v₂|cos(θ)<br />
          = {fmt(vLen)}·{fmt(vLen)}·cos({fmt(angleBetween * RAD, 1)}°)<br />
          <span style={{ color: C.gold }}>= {fmt(dot)}</span>
        </div>
      </div>
    </>
  );
}

interface EigenSidebarProps {
  state: EigenState;
}

function EigenSidebar({ state }: EigenSidebarProps) {
  const { presetKey, setPresetKey, phase, goToPhase } = state;

  const matrix = EIGEN_PRESETS[presetKey].matrix;
  const eigen = eigen2x2(matrix);

  const phaseLabels: { id: EigenPhase; label: string }[] = [
    { id: "before",    label: "Before" },
    { id: "after",     label: "After" },
    { id: "highlight", label: "Highlight" },
  ];

  return (
    <>
      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">Eigenvectors</h2>
        <div className="w-10 h-px bg-[#7a6248] mb-3" />
        <p className="text-sm text-[#6b6b60] leading-relaxed">
          Eigenvectors are the special directions that only scale under a transformation — they don&apos;t rotate.
          <span style={{ color: C.gold }}> Gold vectors</span> don&apos;t change direction.
        </p>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <SectionHeader>Matrix</SectionHeader>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(EIGEN_PRESETS).map(([key, { label }]) => (
            <motion.button
              key={key}
              whileTap={{ scale: 0.95 }}
              onClick={() => { setPresetKey(key); goToPhase("before"); }}
              className={`px-2 py-2 rounded-xl text-xs font-medium border transition-all
                ${presetKey === key
                  ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248]"
                  : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25"
                }`}
            >
              {label}
            </motion.button>
          ))}
        </div>
      </div>

      <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
        <SectionHeader>Steps</SectionHeader>
        <div className="flex gap-2">
          {phaseLabels.map(({ id, label }) => (
            <motion.button
              key={id}
              whileTap={{ scale: 0.95 }}
              onClick={() => goToPhase(id)}
              className={`flex-1 py-2.5 rounded-xl text-xs font-semibold border transition-all
                ${phase === id
                  ? "bg-[#7a6248]/10 border-[#7a6248]/30 text-[#7a6248]"
                  : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25"
                }`}
            >
              {label}
            </motion.button>
          ))}
        </div>
      </div>

      <div className="p-4 sm:p-6">
        <SectionHeader>Eigenvalues & Eigenvectors</SectionHeader>
        {eigen.isReal ? (
          <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] px-4 py-2 space-y-2">
            <div className="flex items-center justify-between border-b border-[#ddddd4] pb-2">
              <span className="text-xs text-[#6b6b60]">λ₁</span>
              <span className="font-mono text-xs font-semibold" style={{ color: C.gold }}>{fmt(eigen.lambda1)}</span>
              <span className="text-xs text-[#6b6b60]">v = ({fmt(eigen.vec1[0])}, {fmt(eigen.vec1[1])})</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#6b6b60]">λ₂</span>
              <span className="font-mono text-xs font-semibold" style={{ color: C.gold }}>{fmt(eigen.lambda2)}</span>
              <span className="text-xs text-[#6b6b60]">v = ({fmt(eigen.vec2[0])}, {fmt(eigen.vec2[1])})</span>
            </div>
          </div>
        ) : (
          <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] p-3 text-xs text-[#6b6b60]">
            Complex eigenvalues (rotation without real eigenvectors)
          </div>
        )}
      </div>
    </>
  );
}

// ── Main component — holds all shared state ───────────────────────────────────

export default function LinearAlgebraVisualizer() {
  const [topic, setTopic] = useState<Topic>("vectors");

  // ── Vectors state ─────────────────────────────────────────────────────────
  const [v1, setV1] = useState<Vec2>([2, 1]);
  const [v2, setV2] = useState<Vec2>([-1, 2]);

  const vectorsState: VectorsState = { v1, v2, setV1, setV2 };

  // ── Transform state ───────────────────────────────────────────────────────
  const [matInputs, setMatInputs] = useState<MatInputs>({ a: "1", b: "0", c: "0", d: "1" });
  const [targetMatrix, setTargetMatrix] = useState<Mat2>([[1, 0], [0, 1]]);
  const [transformT, setTransformT] = useState(0);
  const [transformAnimating, setTransformAnimating] = useState(false);
  const [transformApplied, setTransformApplied] = useState(false);
  const transformRafRef = useRef<number>(0);
  // Keep a ref to matInputs for the applyMatrix callback (avoids stale closure)
  const matInputsRef = useRef(matInputs);
  useEffect(() => { matInputsRef.current = matInputs; }, [matInputs]);

  const applyMatrix = useCallback((overrideMatrix?: Mat2) => {
    let m: Mat2;
    if (overrideMatrix) {
      m = overrideMatrix;
    } else {
      const inputs = matInputsRef.current;
      const a = parseFloat(inputs.a) || 0;
      const b = parseFloat(inputs.b) || 0;
      const c = parseFloat(inputs.c) || 0;
      const d = parseFloat(inputs.d) || 0;
      m = [[a, b], [c, d]];
    }
    setTargetMatrix(m);
    setTransformApplied(true);
    setTransformAnimating(true);
    setTransformT(0);
    cancelAnimationFrame(transformRafRef.current);
    const startTime = performance.now();
    const duration = 700;
    const tick = (now: number) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2;
      setTransformT(eased);
      if (progress < 1) {
        transformRafRef.current = requestAnimationFrame(tick);
      } else {
        setTransformAnimating(false);
      }
    };
    transformRafRef.current = requestAnimationFrame(tick);
  }, []);

  const resetTransform = useCallback(() => {
    cancelAnimationFrame(transformRafRef.current);
    setTransformT(0);
    setTransformApplied(false);
    setTransformAnimating(false);
  }, []);

  // When preset buttons set matInputs, we need the actual matrix values applied, not stale state.
  // We expose a special setter that also records the new inputs in the ref synchronously.
  const setMatInputsAndRef = useCallback((inputs: MatInputs) => {
    matInputsRef.current = inputs;
    setMatInputs(inputs);
  }, []);

  useEffect(() => {
    return () => cancelAnimationFrame(transformRafRef.current);
  }, []);

  const transformState: TransformState = {
    matInputs, setMatInputs: setMatInputsAndRef,
    targetMatrix, t: transformT,
    animating: transformAnimating, applied: transformApplied,
    applyMatrix, resetTransform,
  };

  // ── Dot state ─────────────────────────────────────────────────────────────
  const [angle1, setAngle1] = useState(30);
  const [angle2, setAngle2] = useState(120);

  const dotState: DotState = { angle1, angle2, setAngle1, setAngle2 };

  // ── Eigen state ───────────────────────────────────────────────────────────
  const [eigenPresetKey, setEigenPresetKey] = useState("sym");
  const [eigenPhase, setEigenPhase] = useState<EigenPhase>("before");
  const [eigenAnimT, setEigenAnimT] = useState(0);
  const eigenRafRef = useRef<number>(0);
  const eigenAnimTRef = useRef(0);
  useEffect(() => { eigenAnimTRef.current = eigenAnimT; }, [eigenAnimT]);

  const goToPhase = useCallback((nextPhase: EigenPhase) => {
    cancelAnimationFrame(eigenRafRef.current);
    const startT = eigenAnimTRef.current;
    const targetT = nextPhase === "before" ? 0 : 1;
    const startTime = performance.now();
    const duration = 600;

    setEigenPhase(nextPhase);

    const tick = (now: number) => {
      const p = Math.min((now - startTime) / duration, 1);
      const eased = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      const current = startT + (targetT - startT) * eased;
      eigenAnimTRef.current = current;
      setEigenAnimT(current);
      if (p < 1) eigenRafRef.current = requestAnimationFrame(tick);
    };
    eigenRafRef.current = requestAnimationFrame(tick);
  }, []);

  const setEigenPresetKeyAndReset = useCallback((key: string) => {
    cancelAnimationFrame(eigenRafRef.current);
    setEigenPresetKey(key);
    setEigenPhase("before");
    eigenAnimTRef.current = 0;
    setEigenAnimT(0);
  }, []);

  useEffect(() => {
    return () => cancelAnimationFrame(eigenRafRef.current);
  }, []);

  const eigenState: EigenState = {
    presetKey: eigenPresetKey,
    setPresetKey: setEigenPresetKeyAndReset,
    phase: eigenPhase,
    animT: eigenAnimT,
    goToPhase,
  };

  // ── Render ────────────────────────────────────────────────────────────────

  const renderViz = () => {
    switch (topic) {
      case "vectors":   return <VizVectorAddition state={vectorsState} />;
      case "transform": return <VizTransform state={transformState} />;
      case "dot":       return <VizDot state={dotState} />;
      case "eigen":     return <VizEigen state={eigenState} />;
    }
  };

  const renderSidebar = () => {
    switch (topic) {
      case "vectors":   return <VectorsSidebar state={vectorsState} />;
      case "transform": return <TransformSidebar state={transformState} />;
      case "dot":       return <DotSidebar state={dotState} />;
      case "eigen":     return <EigenSidebar state={eigenState} />;
    }
  };

  return (
    <div
      className="flex flex-col bg-[#f5f5f0] text-[#1a1a1a] overflow-hidden"
      style={{ height: "calc(100vh - 3.5rem)" }}
    >
      <div className="flex-1 flex flex-col lg:flex-row min-h-0">

        {/* ── Main visualization ───────────────────────────────────────────── */}
        <main
          className="order-first lg:order-2 flex-shrink-0 lg:flex-shrink lg:flex-1
                     h-[45vh] lg:h-auto border-b lg:border-b-0 border-[#ddddd4]
                     dot-pattern bg-[#f5f5f0] flex flex-col min-h-0 relative overflow-hidden"
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={topic}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.97 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              className="flex-1 flex flex-col min-h-0 w-full h-full"
            >
              {renderViz()}
            </motion.div>
          </AnimatePresence>

          {/* Warm radial glow at base */}
          <div className="absolute inset-x-0 bottom-0 h-16
                          bg-gradient-to-t from-[#7a6248]/5 to-transparent pointer-events-none" />
        </main>

        {/* ── Sidebar ──────────────────────────────────────────────────────── */}
        <motion.aside
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, delay: 0.1, ease: "easeOut" }}
          className="order-2 lg:order-1 flex-1 lg:flex-none
                     lg:w-80 xl:w-96
                     overflow-y-auto
                     lg:border-r border-[#ddddd4]
                     bg-white"
        >
          {/* Topic picker */}
          <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">
              Topic
            </p>
            <div className="grid grid-cols-2 gap-2">
              {TOPICS.map((t, i) => (
                <motion.button
                  key={t.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04, duration: 0.25 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setTopic(t.id)}
                  className={`px-2 py-2.5 rounded-xl text-xs font-medium transition-all duration-200
                    border ${topic === t.id
                      ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248] shadow-sm"
                      : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]"
                    }`}
                >
                  {t.short}
                </motion.button>
              ))}
            </div>
          </div>

          {/* Topic-specific sidebar content */}
          <AnimatePresence mode="wait">
            <motion.div
              key={topic}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              {renderSidebar()}
            </motion.div>
          </AnimatePresence>
        </motion.aside>

      </div>
    </div>
  );
}
