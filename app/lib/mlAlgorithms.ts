// ─── ML Math Utilities ────────────────────────────────────────────────────────

// ── Activation functions ───────────────────────────────────────────────────────

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function relu(x: number): number {
  return Math.max(0, x);
}

export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

// ── Gradient Descent ──────────────────────────────────────────────────────────

// Loss function: L(θ) = (θ - 2)² + 1
export function gdLoss(theta: number): number {
  return (theta - 2) ** 2 + 1;
}

// Gradient: dL/dθ = 2(θ - 2)
export function gdGradient(theta: number): number {
  return 2 * (theta - 2);
}

export function gdStep(theta: number, lr: number): number {
  return theta - lr * gdGradient(theta);
}

// Points for the loss curve (range: -4 to 8)
export function gdCurvePoints(svgW: number, svgH: number, padding: number): Array<{ x: number; y: number; theta: number }> {
  const pts: Array<{ x: number; y: number; theta: number }> = [];
  const thetaMin = -4;
  const thetaMax = 8;
  const lossMin = 0;
  const lossMax = 40;
  const plotW = svgW - padding * 2;
  const plotH = svgH - padding * 2;
  const steps = 100;
  for (let i = 0; i <= steps; i++) {
    const theta = thetaMin + (i / steps) * (thetaMax - thetaMin);
    const loss = gdLoss(theta);
    const x = padding + ((theta - thetaMin) / (thetaMax - thetaMin)) * plotW;
    const y = padding + plotH - ((loss - lossMin) / (lossMax - lossMin)) * plotH;
    pts.push({ x, y, theta });
  }
  return pts;
}

export function thetaToSvgX(theta: number, svgW: number, padding: number): number {
  const thetaMin = -4;
  const thetaMax = 8;
  const plotW = svgW - padding * 2;
  return padding + ((theta - thetaMin) / (thetaMax - thetaMin)) * plotW;
}

export function lossToSvgY(loss: number, svgH: number, padding: number): number {
  const lossMin = 0;
  const lossMax = 40;
  const plotH = svgH - padding * 2;
  return padding + plotH - ((loss - lossMin) / (lossMax - lossMin)) * plotH;
}

// Tangent line endpoints at a given theta
export function gdTangentLine(
  theta: number,
  svgW: number,
  svgH: number,
  padding: number
): { x1: number; y1: number; x2: number; y2: number } {
  const grad = gdGradient(theta);
  const loss = gdLoss(theta);
  const cx = thetaToSvgX(theta, svgW, padding);
  const cy = lossToSvgY(loss, svgH, padding);
  // In SVG space, a slope unit in theta → unit in x, unit in loss → unit in y (inverted)
  // Convert: dSvgY/dSvgX = (dy/dloss)*(dloss/dtheta)*(dtheta/dSvgX)
  // dSvgX/dtheta = plotW / (thetaMax - thetaMin)
  const thetaRange = 12;
  const lossRange = 40;
  const plotW = svgW - padding * 2;
  const plotH = svgH - padding * 2;
  const svgSlope = grad * (-plotH / lossRange) * (thetaRange / plotW);
  const halfLen = 60;
  const dx = halfLen / Math.sqrt(1 + svgSlope ** 2);
  const dy = svgSlope * dx;
  return { x1: cx - dx, y1: cy - dy, x2: cx + dx, y2: cy + dy };
}

// ── Gradient Descent 2D (3D surface view) ────────────────────────────────────

// L(θ₁, θ₂) = (θ₁ − 2)² + 0.8·(θ₂ − 1.5)² + 0.5  →  min at (2, 1.5), value 0.5
export const GD2D_T1_RANGE: [number, number] = [-2, 6];
export const GD2D_T2_RANGE: [number, number] = [-1, 5];
export const GD2D_MIN: [number, number] = [2, 1.5];
export const GD2D_LOSS_MIN = 0.5;
export const GD2D_LOSS_MAX = 30;

export function gdLoss2D(t1: number, t2: number): number {
  return (t1 - 2) ** 2 + 0.8 * (t2 - 1.5) ** 2 + 0.5;
}

export function gdGradient2D(t1: number, t2: number): [number, number] {
  return [2 * (t1 - 2), 1.6 * (t2 - 1.5)];
}

export function gdStep2D(pos: [number, number], lr: number): [number, number] {
  const [g1, g2] = gdGradient2D(pos[0], pos[1]);
  return [pos[0] - lr * g1, pos[1] - lr * g2];
}

export function gdConverged2D(pos: [number, number]): boolean {
  const [g1, g2] = gdGradient2D(pos[0], pos[1]);
  return g1 * g1 + g2 * g2 < 1e-4;
}

// ── K-Means Clustering ────────────────────────────────────────────────────────

export type Point2D = { x: number; y: number };
export type KMeansState = {
  points: Point2D[];
  centroids: Point2D[];
  assignments: number[];
  k: number;
  converged: boolean;
  iteration: number;
};

export function generateClusteredPoints(count: number): Point2D[] {
  // Generate points in ~3-4 natural clusters
  const clusters = [
    { cx: 0.2, cy: 0.25, spread: 0.1 },
    { cx: 0.75, cy: 0.2, spread: 0.12 },
    { cx: 0.5,  cy: 0.7, spread: 0.11 },
    { cx: 0.15, cy: 0.75, spread: 0.09 },
  ];
  const pts: Point2D[] = [];
  for (let i = 0; i < count; i++) {
    const c = clusters[i % clusters.length];
    pts.push({
      x: Math.max(0.02, Math.min(0.98, c.cx + (Math.random() - 0.5) * c.spread * 2.5)),
      y: Math.max(0.02, Math.min(0.98, c.cy + (Math.random() - 0.5) * c.spread * 2.5)),
    });
  }
  return pts.sort(() => Math.random() - 0.5);
}

export function initializeCentroids(points: Point2D[], k: number): Point2D[] {
  // K-Means++ initialization
  const centroids: Point2D[] = [];
  centroids.push(points[Math.floor(Math.random() * points.length)]);
  for (let c = 1; c < k; c++) {
    const distances = points.map((p) => {
      const minDist = Math.min(...centroids.map((cent) => dist2(p, cent)));
      return minDist;
    });
    const total = distances.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    let chosen = 0;
    for (let i = 0; i < distances.length; i++) {
      r -= distances[i];
      if (r <= 0) { chosen = i; break; }
    }
    centroids.push({ ...points[chosen] });
  }
  return centroids;
}

function dist2(a: Point2D, b: Point2D): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

export function assignPoints(points: Point2D[], centroids: Point2D[]): number[] {
  return points.map((p) => {
    let best = 0;
    let bestDist = Infinity;
    centroids.forEach((c, i) => {
      const d = dist2(p, c);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  });
}

export function updateCentroids(points: Point2D[], assignments: number[], k: number): Point2D[] {
  return Array.from({ length: k }, (_, i) => {
    const members = points.filter((_, idx) => assignments[idx] === i);
    if (members.length === 0) return { x: Math.random(), y: Math.random() };
    return {
      x: members.reduce((s, p) => s + p.x, 0) / members.length,
      y: members.reduce((s, p) => s + p.y, 0) / members.length,
    };
  });
}

export function kMeansStep(state: KMeansState): KMeansState {
  if (state.converged) return state;
  const assignments = assignPoints(state.points, state.centroids);
  const newCentroids = updateCentroids(state.points, assignments, state.k);
  const converged = newCentroids.every((c, i) => dist2(c, state.centroids[i]) < 1e-8);
  return { ...state, assignments, centroids: newCentroids, converged, iteration: state.iteration + 1 };
}

// ── Linear Regression ────────────────────────────────────────────────────────

export type LinRegState = {
  points: Point2D[];
  slope: number;
  intercept: number;
  mse: number;
  iteration: number;
  converged: boolean;
};

export function generateLinearData(count: number): Point2D[] {
  const trueSlope = (Math.random() * 1.4 - 0.4);
  const trueIntercept = Math.random() * 0.3 + 0.1;
  const pts: Point2D[] = [];
  for (let i = 0; i < count; i++) {
    const x = 0.05 + (i / (count - 1)) * 0.9 + (Math.random() - 0.5) * 0.04;
    const y = Math.max(0.02, Math.min(0.98, trueSlope * x + trueIntercept + (Math.random() - 0.5) * 0.18));
    pts.push({ x, y });
  }
  return pts;
}

export function computeMSE(points: Point2D[], slope: number, intercept: number): number {
  const n = points.length;
  const sum = points.reduce((acc, p) => {
    const pred = slope * p.x + intercept;
    return acc + (pred - p.y) ** 2;
  }, 0);
  return sum / n;
}

export function linRegStep(state: LinRegState, lr: number): LinRegState {
  const { points, slope, intercept } = state;
  const n = points.length;
  let dSlope = 0;
  let dIntercept = 0;
  for (const p of points) {
    const err = slope * p.x + intercept - p.y;
    dSlope += err * p.x;
    dIntercept += err;
  }
  dSlope = (2 / n) * dSlope;
  dIntercept = (2 / n) * dIntercept;
  const newSlope = slope - lr * dSlope;
  const newIntercept = intercept - lr * dIntercept;
  const mse = computeMSE(points, newSlope, newIntercept);
  const converged = Math.abs(dSlope) < 1e-6 && Math.abs(dIntercept) < 1e-6;
  return { ...state, slope: newSlope, intercept: newIntercept, mse, iteration: state.iteration + 1, converged };
}

// ── Neural Network Forward Pass ───────────────────────────────────────────────

export type ActivationFn = "relu" | "sigmoid";

export type NNWeights = {
  w1: number[][];  // [3][2] — hidden layer, 3 neurons, 2 inputs each
  b1: number[];    // [3]
  w2: number[][];  // [2][3] — output layer, 2 neurons, 3 inputs each
  b2: number[];    // [2]
};

export type NNActivations = {
  inputs: number[];        // [2]
  hidden: number[];        // [3]
  hiddenPre: number[];     // [3] pre-activation
  outputs: number[];       // [2]
  outputsPre: number[];    // [2] pre-activation
};

export function randomWeights(): NNWeights {
  const rand = () => (Math.random() * 2 - 1) * 0.8;
  return {
    w1: Array.from({ length: 3 }, () => [rand(), rand()]),
    b1: Array.from({ length: 3 }, rand),
    w2: Array.from({ length: 2 }, () => [rand(), rand(), rand()]),
    b2: Array.from({ length: 2 }, rand),
  };
}

export function forwardPass(inputs: number[], weights: NNWeights, fn: ActivationFn): NNActivations {
  const act = fn === "sigmoid" ? sigmoid : relu;
  // Hidden layer
  const hiddenPre = weights.w1.map((row, i) => row.reduce((s, w, j) => s + w * inputs[j], weights.b1[i]));
  const hidden = hiddenPre.map(act);
  // Output layer
  const outputsPre = weights.w2.map((row, i) => row.reduce((s, w, j) => s + w * hidden[j], weights.b2[i]));
  const outputs = outputsPre.map(sigmoid); // always sigmoid at output
  return { inputs, hidden, hiddenPre, outputs, outputsPre };
}
