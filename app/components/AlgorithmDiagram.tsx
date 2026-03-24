"use client";

import { useState, useEffect } from "react";
import { motion, LayoutGroup } from "framer-motion";
import { diagramColors as C } from "../lib/colors";

function usePhase(phases: { id: string; ms: number }[]) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setI(p => (p + 1) % phases.length), phases[i].ms);
    return () => clearTimeout(t);
  }, [i]); // eslint-disable-line
  return phases[i].id;
}

interface Bar { id: number; val: number; color: string }

function Bars({ bars, groupId }: { bars: Bar[]; groupId: string }) {
  const max = Math.max(...bars.map(b => b.val), 1);
  return (
    <LayoutGroup id={groupId}>
      <div className="flex items-end gap-1 h-20 w-full">
        {bars.map(b => (
          <motion.div
            key={b.id}
            layout
            className="flex-1 rounded-t origin-bottom"
            animate={{ backgroundColor: b.color }}
            transition={{ layout: { duration: 0.32, ease: "easeInOut" }, backgroundColor: { duration: 0.18 } }}
            style={{ height: `${(b.val / max) * 100}%`, backgroundColor: b.color }}
          />
        ))}
      </div>
    </LayoutGroup>
  );
}

function Shell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-[#ddddd4] overflow-hidden mb-5">
      <div className="px-4 pt-4 pb-2 bg-[#f5f5f0]">{children}</div>
      <div className="px-4 py-2 bg-[#eeeee8] text-[11px] text-[#6b6b60] font-medium min-h-[30px] flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-[#7a6248] flex-shrink-0 opacity-60" />
        <span>{label || " "}</span>
      </div>
    </div>
  );
}

// ── Bubble ────────────────────────────────────────────────────────────────────
function BubbleDiagram() {
  const PHASES = [
    { id: "idle",    ms: 900 },
    { id: "compare", ms: 650 },
    { id: "swap",    ms: 700 },
    { id: "after",   ms: 800 },
    { id: "reset",   ms: 150 },
  ];
  const phase = usePhase(PHASES);

  const base    = [{ id:0,val:65 },{ id:1,val:80 },{ id:2,val:40 },{ id:3,val:50 },{ id:4,val:90 }];
  const swapped = [{ id:0,val:65 },{ id:2,val:40 },{ id:1,val:80 },{ id:3,val:50 },{ id:4,val:90 }];
  const arr = (phase === "swap" || phase === "after") ? swapped : base;

  const color = (id: number) => {
    if (phase === "compare" && (id===1||id===2)) return C.compare;
    if (phase === "swap"    && (id===1||id===2)) return C.swap;
    if (phase === "after"   && (id===1||id===2)) return C.sorted;
    return C.bar;
  };

  const labels: Record<string,string> = {
    idle:    "Scanning for out-of-order adjacent pairs",
    compare: "80 > 40 — these two are out of order",
    swap:    "Swapping the pair",
    after:   "40 and 80 are now in order ✓",
    reset:   "",
  };

  return (
    <Shell label={labels[phase] ?? ""}>
      <Bars groupId="bubble" bars={arr.map(({id,val}) => ({ id, val, color: color(id) }))} />
    </Shell>
  );
}

// ── Selection ─────────────────────────────────────────────────────────────────
function SelectionDiagram() {
  const vals = [72, 18, 55, 34, 91];
  const PHASES = [
    { id:"idle",  ms:800 },{ id:"s0", ms:360 },{ id:"s1", ms:360 },
    { id:"s2",    ms:360 },{ id:"s3", ms:360 },{ id:"s4", ms:360 },
    { id:"found", ms:600 },{ id:"swap", ms:700 },{ id:"done", ms:850 },
    { id:"reset", ms:150 },
  ];
  const phase = usePhase(PHASES);

  const minAt:  Record<string,number> = { idle:-1,s0:0,s1:1,s2:1,s3:1,s4:1,found:1,swap:1,done:1,reset:-1 };
  const scanAt: Record<string,number> = { idle:-1,s0:0,s1:1,s2:2,s3:3,s4:4,found:4,swap:-1,done:-1,reset:-1 };
  const isSwapped = phase === "swap" || phase === "done";
  const order = isSwapped ? [1,0,2,3,4] : [0,1,2,3,4];

  const color = (id: number) => {
    if (phase==="done"  && id===1) return C.sorted;
    if (isSwapped       && (id===0||id===1)) return C.swap;
    if (id===minAt[phase])  return C.compare;
    if (id===scanAt[phase]) return C.pivot;
    return C.bar;
  };

  const labels: Record<string,string> = {
    idle:"Finding the minimum of the unsorted portion", s0:"Current min: 72",
    s1:"18 < 72 — new minimum!", s2:"55 > 18 — min unchanged",
    s3:"34 > 18 — min unchanged", s4:"91 > 18 — minimum is 18",
    found:"Minimum found: 18", swap:"Swapping 18 to the sorted front",
    done:"18 is in its final position ✓", reset:"",
  };

  return (
    <Shell label={labels[phase] ?? ""}>
      <Bars groupId="selection" bars={order.map(id => ({ id, val:vals[id], color:color(id) }))} />
    </Shell>
  );
}

// ── Insertion ─────────────────────────────────────────────────────────────────
function InsertionDiagram() {
  const vals: Record<number,number> = { 0:28, 1:55, 2:78, 3:40, 4:62 };
  const PHASES = [
    { id:"idle",  ms:800 },{ id:"pick",  ms:600 },
    { id:"cmp2",  ms:450 },{ id:"sft2",  ms:380 },
    { id:"cmp1",  ms:450 },{ id:"sft1",  ms:380 },
    { id:"cmp0",  ms:450 },{ id:"place", ms:600 },
    { id:"done",  ms:850 },{ id:"reset", ms:150 },
  ];
  const phase = usePhase(PHASES);

  const orderMap: Record<string,number[]> = {
    idle:[0,1,2,3,4], pick:[0,1,2,3,4], cmp2:[0,1,2,3,4],
    sft2:[0,1,3,2,4], cmp1:[0,1,3,2,4],
    sft1:[0,3,1,2,4], cmp0:[0,3,1,2,4],
    place:[0,3,1,2,4], done:[0,3,1,2,4], reset:[0,1,2,3,4],
  };

  const color = (id: number) => {
    if (phase==="done"  && id===3) return C.sorted;
    if (["sft2","sft1","place"].includes(phase) && id===3) return C.swap;
    if (id===3) return C.compare;
    if (phase==="cmp2" && id===2) return C.pivot;
    if (phase==="cmp1" && id===1) return C.pivot;
    if (phase==="cmp0" && id===0) return C.pivot;
    if ([0,1,2].includes(id) && !["idle","reset"].includes(phase)) return "#b8a8d8";
    return C.bar;
  };

  const order = orderMap[phase] ?? [0,1,2,3,4];
  const labels: Record<string,string> = {
    idle:"Inserting next element into the sorted portion", pick:"Key element: 40",
    cmp2:"78 > 40 — shift 78 right", sft2:"78 shifted",
    cmp1:"55 > 40 — shift 55 right", sft1:"55 shifted",
    cmp0:"28 < 40 — correct position found", place:"Inserting 40",
    done:"40 is in its sorted position ✓", reset:"",
  };

  return (
    <Shell label={labels[phase] ?? ""}>
      <Bars groupId="insertion" bars={order.map(id => ({ id, val:vals[id], color:color(id) }))} />
    </Shell>
  );
}

// ── Merge ─────────────────────────────────────────────────────────────────────
function MergeDiagram() {
  const vals: Record<number,number> = { 0:20, 1:48, 2:73, 3:31, 4:55, 5:82 };
  const PHASES = [
    { id:"two-arrays", ms:1000 },
    { id:"cmp-1",      ms:550  },{ id:"take-20",  ms:450 },
    { id:"cmp-2",      ms:550  },{ id:"take-31",  ms:450 },
    { id:"cmp-3",      ms:550  },{ id:"take-48",  ms:450 },
    { id:"done",       ms:900  },{ id:"reset",     ms:200 },
  ];
  const phase = usePhase(PHASES);

  const placedMap: Record<string,number[]> = {
    "two-arrays":[], "cmp-1":[],
    "take-20":[0], "cmp-2":[0],
    "take-31":[0,3], "cmp-3":[0,3],
    "take-48":[0,3,1],
    done:[0,3,1,4,2,5], reset:[],
  };

  const placed = placedMap[phase] ?? [];
  const remaining = [0,1,2,3,4,5].filter(id => !placed.includes(id));
  const order = [...placed, ...remaining];
  const leftFront  = remaining.find(id => id < 3) ?? -1;
  const rightFront = remaining.find(id => id >= 3) ?? -1;
  const isCmp = ["cmp-1","cmp-2","cmp-3"].includes(phase);

  const color = (id: number) => {
    if (placed.includes(id)) return C.sorted;
    if (isCmp && id===leftFront)  return C.compare;
    if (isCmp && id===rightFront) return C.left;
    return id < 3 ? C.left : C.right;
  };

  const labels: Record<string,string> = {
    "two-arrays":"Two sorted halves ready to merge",
    "cmp-1":"Compare 20 (left) vs 31 (right)",
    "take-20":"20 < 31 — take 20",
    "cmp-2":"Compare 31 (right) vs 48 (left)",
    "take-31":"31 < 48 — take 31",
    "cmp-3":"Compare 48 (left) vs 55 (right)",
    "take-48":"48 < 55 — take 48",
    done:"Merge complete ✓", reset:"",
  };

  return (
    <Shell label={labels[phase] ?? ""}>
      <Bars groupId="merge" bars={order.map(id => ({ id, val:vals[id], color:color(id) }))} />
    </Shell>
  );
}

// ── Quick ─────────────────────────────────────────────────────────────────────
function QuickDiagram() {
  const vals: Record<number,number> = { 0:64, 1:27, 2:85, 3:41, 4:93, 5:52 };
  const PHASES = [
    { id:"idle",      ms:1000 },
    { id:"scan-0",    ms:400  },{ id:"scan-1", ms:400 },
    { id:"scan-2",    ms:400  },{ id:"scan-3", ms:400 },
    { id:"scan-4",    ms:400  },
    { id:"partition", ms:700  },{ id:"done",   ms:900 },
    { id:"reset",     ms:200  },
  ];
  const phase = usePhase(PHASES);

  const finalOrder = [1,3,5,0,2,4];
  const initOrder  = [0,1,2,3,4,5];
  const orderMap: Record<string,number[]> = {
    idle:initOrder, "scan-0":initOrder, "scan-1":initOrder,
    "scan-2":initOrder, "scan-3":initOrder, "scan-4":initOrder,
    partition:finalOrder, done:finalOrder, reset:initOrder,
  };

  const leftSoFar:  Record<string,number[]> = {
    idle:[], "scan-0":[], "scan-1":[1], "scan-2":[1], "scan-3":[1,3], "scan-4":[1,3],
    partition:[1,3], done:[1,3], reset:[],
  };
  const rightSoFar: Record<string,number[]> = {
    idle:[], "scan-0":[0], "scan-1":[0], "scan-2":[0,2], "scan-3":[0,2], "scan-4":[0,2,4],
    partition:[0,2,4], done:[0,2,4], reset:[],
  };
  const scanId: Record<string,number> = {
    "scan-0":0,"scan-1":1,"scan-2":2,"scan-3":3,"scan-4":4,
  };

  const color = (id: number) => {
    if (id===5) return C.pivot;
    if (leftSoFar[phase]?.includes(id))  return C.compare;
    if (rightSoFar[phase]?.includes(id)) return C.swap;
    if (scanId[phase]===id) return C.right;
    return C.bar;
  };

  const order = orderMap[phase] ?? initOrder;
  const labels: Record<string,string> = {
    idle:"Pivot chosen: last element (52)",
    "scan-0":"64 > 52 → right of pivot",
    "scan-1":"27 < 52 → left of pivot",
    "scan-2":"85 > 52 → right of pivot",
    "scan-3":"41 < 52 → left of pivot",
    "scan-4":"93 > 52 → right of pivot",
    partition:"Pivot 52 moves to its final position",
    done:"Left < 52 ≤ Right — pivot is sorted ✓",
    reset:"",
  };

  return (
    <Shell label={labels[phase] ?? ""}>
      <Bars groupId="quick" bars={order.map(id => ({ id, val:vals[id], color:color(id) }))} />
    </Shell>
  );
}

// ── Heap (SVG tree) ───────────────────────────────────────────────────────────
const NODE_XY = [
  { x:140, y:22 }, { x:78,  y:62 }, { x:202, y:62 },
  { x:42,  y:102 },{ x:114, y:102 },{ x:166, y:102 },{ x:238, y:102 },
];
const EDGES = [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]];
const HEAP  = [91,75,63,54,68,40,29];

function HeapDiagram() {
  const PHASES = [
    { id:"idle",     ms:1000 },
    { id:"max",      ms:700  },
    { id:"extract",  ms:750  },
    { id:"heapify",  ms:750  },
    { id:"done",     ms:900  },
    { id:"reset",    ms:200  },
  ];
  const phase = usePhase(PHASES);

  const nodeColor = (idx: number): string => {
    if (phase==="max"     && idx===0) return C.pivot;
    if (phase==="extract" && idx===0) return C.swap;
    if (phase==="heapify" && idx===0) return C.compare;
    if (phase==="heapify" && idx===1) return C.pivot;
    if (phase==="done"    && idx===0) return C.sorted;
    return C.bar;
  };

  const displayVal = (idx: number): number => {
    if (phase==="heapify" && idx===0) return HEAP[6];
    if (phase==="heapify" && idx===6) return HEAP[0];
    if (phase==="done"    && idx===0) return HEAP[1];
    if (phase==="done"    && idx===1) return 29;
    return HEAP[idx];
  };

  const showSorted = phase==="extract" || phase==="heapify" || phase==="done";

  const labels: Record<string,string> = {
    idle:    "Max-heap: every parent ≥ its children",
    max:     "Root is always the maximum (91)",
    extract: "Extract 91 — it goes to its sorted position",
    heapify: "Last node (29) becomes root, then sinks down",
    done:    "75 rises to root after heapify ✓",
    reset:   "",
  };

  return (
    <Shell label={labels[phase] ?? ""}>
      <svg viewBox="0 0 280 124" className="w-full" style={{ height: 116 }}>
        {EDGES.map(([a,b]) => (
          <line key={`${a}-${b}`}
            x1={NODE_XY[a].x} y1={NODE_XY[a].y}
            x2={NODE_XY[b].x} y2={NODE_XY[b].y}
            stroke="#ddddd4" strokeWidth={1.5}
          />
        ))}
        {NODE_XY.map(({ x,y }, idx) => (
          <motion.g key={idx}
            animate={{ opacity: phase==="extract" && idx===0 ? 0.25 : 1 }}
            transition={{ duration: 0.3 }}
          >
            <motion.circle cx={x} cy={y} r={16}
              animate={{ fill: nodeColor(idx) }}
              initial={{ fill: C.bar }}
              transition={{ duration: 0.25 }}
              stroke="white" strokeWidth={2}
            />
            <text x={x} y={y} textAnchor="middle" dominantBaseline="central"
              fontSize={10} fontWeight="600"
              fontFamily="var(--font-geist-mono, monospace)"
              fill="white" style={{ pointerEvents:"none" }}>
              {displayVal(idx)}
            </text>
          </motion.g>
        ))}
        {/* Extracted node floating to the right */}
        {showSorted && (
          <motion.g initial={{ opacity:0, y:-8 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.35 }}>
            <circle cx={264} cy={22} r={16} fill={C.sorted} stroke="white" strokeWidth={2} />
            <text x={264} y={22} textAnchor="middle" dominantBaseline="central"
              fontSize={10} fontWeight="600"
              fontFamily="var(--font-geist-mono, monospace)" fill="white">
              91
            </text>
          </motion.g>
        )}
      </svg>
    </Shell>
  );
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
const DIAGRAMS: Record<string, React.FC> = {
  bubble: BubbleDiagram, selection: SelectionDiagram, insertion: InsertionDiagram,
  merge:  MergeDiagram,  quick:     QuickDiagram,     heap:      HeapDiagram,
};

export default function AlgorithmDiagram({ algorithm }: { algorithm: string }) {
  const Diagram = DIAGRAMS[algorithm];
  return Diagram ? <Diagram /> : null;
}
