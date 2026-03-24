"use client";

import { useEffect, useRef, useState } from "react";
import { motion, LayoutGroup } from "framer-motion";
import { algorithmFunctions, algorithmInfo, type SortStep } from "../lib/algorithms";
import { sortColors } from "../lib/colors";
import { generateArray, barColor } from "../lib/sortingUtils";

const ALGORITHMS = Object.keys(algorithmInfo);
const N = 14;
const STEP_MS = 60;
const PAUSE_MS = 1200;

export default function HeroAnimation() {
  const [bars, setBars]       = useState<number[]>(() => generateArray(N, 10, 96));
  const [step, setStep]       = useState<SortStep | null>(null);
  const [label, setLabel]     = useState(algorithmInfo["bubble"].name);

  const stepsRef    = useRef<SortStep[]>([]);
  const stepIdxRef  = useRef(0);
  const algoIdxRef  = useRef(0);
  const timerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);

  function tick() {
    timerRef.current = setTimeout(() => {
      const i = stepIdxRef.current;
      const steps = stepsRef.current;

      if (i >= steps.length) {
        timerRef.current = setTimeout(() => {
          const nextAlgoIdx = (algoIdxRef.current + 1) % ALGORITHMS.length;
          algoIdxRef.current = nextAlgoIdx;
          const algoKey = ALGORITHMS[nextAlgoIdx];

          const nextArr = generateArray(N, 10, 96);
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
      setBars([...s.array]);
      setStep(s);
      stepIdxRef.current = i + 1;
      tick();
    }, STEP_MS);
  }

  useEffect(() => {
    const arr = generateArray(N, 10, 96);
    const algoKey = ALGORITHMS[0];
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

  // Build highlight sets from the current step
  const comparing = new Set(step?.comparing ?? []);
  const swapping  = new Set(step?.swapping ?? []);
  const sorted    = new Set(step?.sorted ?? []);
  const pivot     = step?.pivot;

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
          {bars.map((val, pos) => {
            const id = step?.ids?.[pos] ?? pos;
            const heightPct = (val / maxVal) * 100;
            const color = barColor(pos, comparing, swapping, sorted, pivot);
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
            <span className="w-2 h-2 rounded-sm" style={{ background: sortColors[k] }} />
            <span className="text-[10px] font-mono capitalize" style={{ color: "var(--muted)" }}>
              {k}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
