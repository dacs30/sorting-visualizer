"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence, LayoutGroup } from "framer-motion";
import { algorithmInfo, algorithmFunctions, type SortStep } from "../lib/algorithms";
import AlgorithmDiagram from "./AlgorithmDiagram";

const ALGORITHMS = Object.keys(algorithmInfo);

// Light-theme bar palette
const C = {
  unsorted:  "#a090c8",
  comparing: "#3b90cc",
  swapping:  "#c86030",
  pivot:     "#b89020",
  sorted:    "#3a9a50",
} as const;

// Generate N unique heights evenly spread across [8, 96]
function generateArray(size: number): number[] {
  return Array.from({ length: size }, (_, i) => Math.round(8 + (i / (size - 1)) * 88))
    .sort(() => Math.random() - 0.5);
}

function barColor(
  i: number,
  comparing: Set<number>,
  swapping: Set<number>,
  sorted: Set<number>,
  pivot: number | undefined,
): string {
  if (sorted.has(i))    return C.sorted;
  if (swapping.has(i))  return C.swapping;
  if (pivot === i)      return C.pivot;
  if (comparing.has(i)) return C.comparing;
  return C.unsorted;
}

// ─── tiny sub-components ──────────────────────────────────────────────────────

function ComplexityRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[#ddddd4] last:border-0">
      <span className="text-xs text-[#6b6b60]">{label}</span>
      <span className={`font-mono text-xs font-semibold ${color}`}>{value}</span>
    </div>
  );
}

function Dot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: color }} />
      <span className="text-xs text-[#6b6b60]">{label}</span>
    </div>
  );
}

// ─── main ─────────────────────────────────────────────────────────────────────

export default function SortingVisualizer() {
  const [algorithm, setAlgorithm]     = useState("bubble");
  const [arraySize, setArraySize]     = useState(40);
  const [speed, setSpeed]             = useState(50);
  const [array, setArray]             = useState<number[]>(() => generateArray(40));
  const [currentStep, setCurrentStep] = useState<SortStep | null>(null);
  const [isRunning, setIsRunning]     = useState(false);
  const [isDone, setIsDone]           = useState(false);
  const [stepIndex, setStepIndex]     = useState(0);

  // Stable ID per value: valueToId[value] = original index (unique values ⇒ 1-to-1)
  const valueToId = useRef<Map<number, number>>(new Map());

  const stepsRef     = useRef<SortStep[]>([]);
  const intervalRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isRunningRef = useRef(false);

  const info = algorithmInfo[algorithm];

  // ── reset ──────────────────────────────────────────────────────────────────
  const reset = useCallback((newArr?: number[]) => {
    if (intervalRef.current) clearTimeout(intervalRef.current);
    isRunningRef.current = false;
    const arr = newArr ?? generateArray(arraySize);
    // Rebuild the value→id map
    const m = new Map<number, number>();
    arr.forEach((v, i) => m.set(v, i));
    valueToId.current = m;
    setArray(arr);
    setCurrentStep(null);
    setIsRunning(false);
    setIsDone(false);
    setStepIndex(0);
    stepsRef.current = [];
  }, [arraySize]);

  useEffect(() => { reset(); }, [algorithm, arraySize]); // eslint-disable-line

  // ── step loop ──────────────────────────────────────────────────────────────
  const runSteps = useCallback((steps: SortStep[], idx: number) => {
    if (idx >= steps.length) {
      isRunningRef.current = false;
      setIsRunning(false);
      setIsDone(true);
      setCurrentStep(steps[steps.length - 1]);
      return;
    }
    setCurrentStep(steps[idx]);
    setStepIndex(idx);
    const delay = Math.max(1, 201 - speed * 2);
    intervalRef.current = setTimeout(() => {
      if (!isRunningRef.current) return;
      runSteps(steps, idx + 1);
    }, delay);
  }, [speed]);

  const handleStart = () => {
    if (isRunning) {
      isRunningRef.current = false;
      if (intervalRef.current) clearTimeout(intervalRef.current);
      setIsRunning(false);
      return;
    }
    if (isDone) return;
    isRunningRef.current = true;
    setIsRunning(true);
    if (stepsRef.current.length === 0) {
      stepsRef.current = algorithmFunctions[algorithm](array);
    }
    runSteps(stepsRef.current, stepIndex);
  };

  // ── derived display state ──────────────────────────────────────────────────
  const displayArray = currentStep?.array ?? array;
  const comparing    = new Set(currentStep?.comparing ?? []);
  const swapping     = new Set(currentStep?.swapping ?? []);
  const sorted       = new Set(currentStep?.sorted ?? []);
  const pivot        = currentStep?.pivot;
  const maxVal       = Math.max(...displayArray, 1);
  const progress     = stepsRef.current.length ? Math.round((stepIndex / stepsRef.current.length) * 100) : 0;

  // Per-step layout transition speed — cap at 120 ms so bars always look snappy
  const stepDelay       = Math.max(1, 201 - speed * 2);
  const layoutDuration  = Math.min(stepDelay * 0.9 / 1000, 0.12);

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div className="h-screen flex flex-col bg-[#f5f5f0] text-[#1a1a1a] overflow-hidden">

      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
        className="flex-shrink-0 border-b border-[#ddddd4] px-5 sm:px-8 py-3 sm:py-4
                   bg-[#f5f5f0]/90 backdrop-blur-md z-10 flex items-center gap-4"
      >
        <div className="flex-1 min-w-0">
          <h1 className="font-serif text-lg sm:text-2xl font-semibold text-[#1a1a1a] leading-tight truncate">
            Sorting Visualizer
          </h1>
          <p className="text-[#6b6b60] text-xs mt-0.5 hidden sm:block">
            Watch algorithms sort in real time — and understand how they work
          </p>
        </div>

        <AnimatePresence>
          {(isRunning || isDone) && (
            <motion.div
              key="status"
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85 }}
              className="flex-shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-full
                         bg-white border border-[#ddddd4] shadow-sm text-xs"
            >
              {isRunning ? (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-[#b89020] animate-pulse" />
                  <span className="text-[#b89020] font-medium">{progress}%</span>
                </>
              ) : (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-[#3a9a50]" />
                  <span className="text-[#3a9a50] font-medium">Sorted</span>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.header>

      {/* Body */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0">

        {/* ── Visualizer ──────────────────────────────────────────────────── */}
        <main
          className="order-first lg:order-2 flex-shrink-0 lg:flex-shrink lg:flex-1
                     h-[38vh] sm:h-[44vh] lg:h-auto
                     border-b lg:border-b-0 border-[#ddddd4]
                     dot-pattern bg-[#f5f5f0]
                     p-4 sm:p-5 flex flex-col min-h-0 relative overflow-hidden"
        >
          {/* Subtle warm radial glow at base */}
          <div className="absolute inset-x-0 bottom-0 h-24
                          bg-gradient-to-t from-[#7a6248]/6 to-transparent pointer-events-none" />

          {/* Idle hint */}
          <AnimatePresence>
            {!isRunning && !isDone && stepIndex === 0 && (
              <motion.p
                key="hint"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute top-3 left-1/2 -translate-x-1/2 whitespace-nowrap
                           text-[10px] tracking-widest uppercase text-[#6b6b60]/50
                           pointer-events-none select-none"
              >
                Press Sort to begin
              </motion.p>
            )}
          </AnimatePresence>

          {/*
            Bars keyed by stable ID (original index of each value).
            When elements swap, their DOM order swaps → Framer Motion `layout`
            animates each bar sliding to its new horizontal position.
          */}
          <LayoutGroup>
            <div className="flex-1 flex items-end gap-px min-h-0 relative z-10">
              {displayArray.map((val, pos) => {
                const id    = valueToId.current.get(val) ?? pos;
                const color = barColor(pos, comparing, swapping, sorted, pivot);
                const hPct  = (val / maxVal) * 100;
                const isActive = comparing.has(pos) || swapping.has(pos) || pivot === pos;

                return (
                  <motion.div
                    key={id}
                    layout
                    layoutId={`bar-${id}`}
                    className="flex-1 rounded-t origin-bottom"
                    animate={{
                      backgroundColor: color,
                      scaleY: isActive ? 1.05 : 1,
                      opacity: sorted.has(pos) ? 1 : 0.82,
                    }}
                    transition={{
                      layout:          { duration: layoutDuration, ease: "easeInOut" },
                      backgroundColor: { duration: layoutDuration },
                      scaleY:          { duration: 0.07, ease: "easeOut" },
                      opacity:         { duration: 0.12 },
                    }}
                    style={{ height: `${hPct}%`, backgroundColor: color }}
                    title={`${val}`}
                  />
                );
              })}
            </div>
          </LayoutGroup>

          {/* Progress track */}
          <div className="flex-shrink-0 mt-2 h-0.5 rounded-full bg-[#ddddd4] overflow-hidden">
            <motion.div
              className="h-full rounded-full bg-[#7a6248]"
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.1 }}
            />
          </div>
        </main>

        {/* ── Sidebar ─────────────────────────────────────────────────────── */}
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

          {/* Algorithm picker */}
          <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">
              Algorithm
            </p>
            <div className="grid grid-cols-3 lg:grid-cols-2 gap-2">
              {ALGORITHMS.map((alg, i) => (
                <motion.button
                  key={alg}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04, duration: 0.25 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => { if (!isRunning) setAlgorithm(alg); }}
                  disabled={isRunning}
                  className={`px-2 py-2.5 rounded-xl text-xs font-medium transition-all duration-200
                    border ${algorithm === alg
                      ? "bg-[#7a6248]/10 border-[#7a6248]/35 text-[#7a6248] shadow-sm"
                      : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]"
                    } disabled:opacity-40 disabled:cursor-not-allowed`}
                >
                  {algorithmInfo[alg].name.replace(" Sort", "")}
                </motion.button>
              ))}
            </div>
          </div>

          {/* Sliders */}
          <div className="p-4 sm:p-6 border-b border-[#ddddd4] space-y-5">
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">Array Size</span>
                <span className="text-xs text-[#7a6248] font-mono font-semibold">{arraySize}</span>
              </div>
              <input
                type="range" min={10} max={80} value={arraySize}
                onChange={(e) => { if (!isRunning) setArraySize(Number(e.target.value)); }}
                disabled={isRunning}
                className="w-full"
              />
            </div>
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60]">Speed</span>
                <span className="text-xs text-[#7a6248] font-mono font-semibold">
                  {speed < 34 ? "Slow" : speed < 67 ? "Medium" : "Fast"}
                </span>
              </div>
              <input
                type="range" min={1} max={100} value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          {/* Buttons */}
          <div className="p-4 sm:p-6 border-b border-[#ddddd4] flex gap-2">
            {[
              {
                label: isRunning ? "Pause" : isDone ? "Done ✓" : stepIndex > 0 ? "Resume" : "Sort",
                onClick: handleStart,
                disabled: isDone,
                active: true,
              },
              { label: "Reset", onClick: () => reset(), disabled: false, active: false },
              { label: "New",   onClick: () => reset(generateArray(arraySize)), disabled: isRunning, active: false },
            ].map(({ label, onClick, disabled, active }) => (
              <motion.button
                key={label}
                whileTap={{ scale: 0.95 }}
                onClick={onClick}
                disabled={disabled}
                className={`flex-1 py-3 rounded-xl font-semibold text-sm transition-all duration-200 border
                  ${active && !isDone
                    ? isRunning
                      ? "bg-[#c86030]/10 border-[#c86030]/30 text-[#c86030] hover:bg-[#c86030]/18"
                      : "bg-[#7a6248]/10 border-[#7a6248]/30 text-[#7a6248] hover:bg-[#7a6248]/18 shadow-sm"
                    : "bg-[#f5f5f0] border-[#ddddd4] text-[#6b6b60] hover:border-[#7a6248]/25 hover:text-[#1a1a1a]"
                  } disabled:opacity-35 disabled:cursor-not-allowed`}
              >
                {label}
              </motion.button>
            ))}
          </div>

          {/* Legend */}
          <div className="p-4 sm:p-6 border-b border-[#ddddd4]">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">Legend</p>
            <div className="grid grid-cols-3 sm:grid-cols-2 lg:grid-cols-2 gap-y-2.5 gap-x-4">
              <Dot color={C.unsorted}  label="Unsorted"  />
              <Dot color={C.comparing} label="Comparing" />
              <Dot color={C.swapping}  label="Swapping"  />
              <Dot color={C.pivot}     label="Pivot"     />
              <Dot color={C.sorted}    label="Sorted"    />
            </div>
          </div>

          {/* Algorithm info — animates on algorithm change */}
          <div className="p-4 sm:p-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={algorithm}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
              >
                <h2 className="font-serif text-lg font-semibold text-[#1a1a1a] mb-1">{info.name}</h2>
                <div className="w-10 h-px bg-[#7a6248] mb-3" />

                <AlgorithmDiagram algorithm={algorithm} />

                <p className="text-sm text-[#6b6b60] leading-relaxed mb-5">{info.description}</p>

                <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-3">
                  How it works
                </p>
                <ol className="space-y-2.5 mb-5">
                  {info.howItWorks.map((step, i) => (
                    <motion.li
                      key={i}
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05, duration: 0.22 }}
                      className="flex gap-3 text-sm text-[#6b6b60]"
                    >
                      <span className="flex-shrink-0 w-5 h-5 rounded-full
                                       bg-[#7a6248]/10 border border-[#7a6248]/25
                                       text-[#7a6248] flex items-center justify-center
                                       text-[10px] font-bold">
                        {i + 1}
                      </span>
                      <span className="leading-relaxed">{step}</span>
                    </motion.li>
                  ))}
                </ol>

                <p className="text-[10px] font-semibold uppercase tracking-widest text-[#6b6b60] mb-2">
                  Complexity
                </p>
                <div className="rounded-xl bg-[#f5f5f0] border border-[#ddddd4] px-4 py-1 mb-2">
                  <ComplexityRow label="Best case"  value={info.timeComplexity.best}    color="text-[#3a9a50]" />
                  <ComplexityRow label="Average"    value={info.timeComplexity.average} color="text-[#b89020]" />
                  <ComplexityRow label="Worst case" value={info.timeComplexity.worst}   color="text-[#c86030]" />
                  <ComplexityRow label="Space"      value={info.spaceComplexity}        color="text-[#3b90cc]" />
                  <div className="flex items-center justify-between pt-1.5">
                    <span className="text-xs text-[#6b6b60]">Stable</span>
                    <span className={`text-xs font-semibold ${info.stable ? "text-[#3a9a50]" : "text-[#c86030]"}`}>
                      {info.stable ? "Yes" : "No"}
                    </span>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

        </motion.aside>
      </div>
    </div>
  );
}
