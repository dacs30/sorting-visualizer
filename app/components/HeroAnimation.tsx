"use client";

import { useEffect, useRef, useState } from "react";
import { motion, LayoutGroup } from "framer-motion";
import { algorithmFunctions, algorithmInfo, type SortStep } from "../lib/algorithms";

const C = {
  unsorted:  "#a090c8",
  comparing: "#3b90cc",
  swapping:  "#c86030",
  sorted:    "#3a9a50",
} as const;

const ALGORITHMS = ["bubble", "merge", "quick", "insertion", "selection", "heap"];
const N = 14;
const STEP_MS = 60;
const PAUSE_MS = 1200;

function generateArray(): number[] {
  return Array.from({ length: N }, (_, i) => Math.round(10 + (i / (N - 1)) * 86))
    .sort(() => Math.random() - 0.5);
}

function barColor(i: number, step: SortStep): string {
  if (step.sorted.includes(i))    return C.sorted;
  if (step.swapping.includes(i))  return C.swapping;
  if (step.pivot === i)           return C.swapping;
  if (step.comparing.includes(i)) return C.comparing;
  return C.unsorted;
}

export default function HeroAnimation() {
  const [bars, setBars]   = useState<number[]>(() => generateArray());
  const [step, setStep]   = useState<SortStep | null>(null);
  const [label, setLabel] = useState(algorithmInfo["bubble"].name);

  // All mutable loop state lives in refs to avoid stale closures
  const valueToId   = useRef<Map<number, number>>(new Map());
  const stepsRef    = useRef<SortStep[]>([]);
  const stepIdxRef  = useRef(0);
  const algoIdxRef  = useRef(0);
  const timerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);

  function initValueMap(arr: number[]) {
    const m = new Map<number, number>();
    arr.forEach((v, i) => m.set(v, i));
    valueToId.current = m;
  }

  function tick() {
    timerRef.current = setTimeout(() => {
      const i = stepIdxRef.current;
      const steps = stepsRef.current;

      if (i >= steps.length) {
        // Sort complete — pause then restart with next algorithm
        timerRef.current = setTimeout(() => {
          const nextAlgoIdx = (algoIdxRef.current + 1) % ALGORITHMS.length;
          algoIdxRef.current = nextAlgoIdx;
          const algoKey = ALGORITHMS[nextAlgoIdx];

          const nextArr = generateArray();
          initValueMap(nextArr);
          stepsRef.current = algorithmFunctions[algoKey](nextArr);
          stepIdxRef.current = 0;

          setBars(nextArr);
          setStep(null);
          setLabel(algorithmInfo[algoKey].name);
          tick();
        }, PAUSE_MS);
        return;
      }

      const s = steps[i];
      // Keep valueToId in sync with swaps so bar DOM identity follows value
      if (s.swapping.length === 2) {
        const [a, b] = s.swapping;
        const idA = valueToId.current.get(s.array[a]);
        const idB = valueToId.current.get(s.array[b]);
        if (idA !== undefined && idB !== undefined) {
          valueToId.current.set(s.array[a], idB);
          valueToId.current.set(s.array[b], idA);
        }
      }

      setBars([...s.array]);
      setStep(s);
      stepIdxRef.current = i + 1;
      tick();
    }, STEP_MS);
  }

  useEffect(() => {
    const arr = generateArray();
    const algoKey = ALGORITHMS[0];
    initValueMap(arr);
    stepsRef.current = algorithmFunctions[algoKey](arr);
    stepIdxRef.current = 0;
    algoIdxRef.current = 0;
    setBars(arr);
    setLabel(algorithmInfo[algoKey].name);
    tick();

    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const maxVal = Math.max(...bars);

  return (
    <div
      className="rounded-2xl border p-5 flex flex-col gap-4 w-full select-none"
      style={{ background: "var(--surface)", borderColor: "var(--border)", minWidth: 260 }}
    >
      {/* Label row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
          live preview
        </span>
        <span
          className="text-xs font-mono px-2 py-0.5 rounded-full"
          style={{ background: "var(--surface2)", color: "var(--primary)" }}
        >
          {label}
        </span>
      </div>

      {/* Bars */}
      <div className="flex items-end gap-[3px]" style={{ height: 140 }}>
        <LayoutGroup id="hero-bars">
          {bars.map((val) => {
            const id = valueToId.current.get(val) ?? val;
            const heightPct = (val / maxVal) * 100;
            const color = step ? barColor(bars.indexOf(val), step) : C.unsorted;
            return (
              <motion.div
                key={id}
                layout
                layoutId={String(id)}
                className="flex-1 rounded-t-sm"
                animate={{
                  height: `${heightPct}%`,
                  backgroundColor: color,
                }}
                transition={{
                  layout: { duration: STEP_MS / 1000, ease: "easeInOut" },
                  backgroundColor: { duration: 0.08 },
                  height: { duration: STEP_MS / 1000, ease: "easeInOut" },
                }}
              />
            );
          })}
        </LayoutGroup>
      </div>

      {/* Color legend */}
      <div className="flex items-center gap-3 flex-wrap">
        {(["comparing", "swapping", "sorted"] as const).map((k) => (
          <div key={k} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm" style={{ background: C[k] }} />
            <span className="text-[10px] font-mono capitalize" style={{ color: "var(--muted)" }}>
              {k}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
