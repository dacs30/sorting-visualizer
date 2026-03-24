// ── Linear Algebra Math Utilities ─────────────────────────────────────────────

export type Vec2 = [number, number];
export type Mat2 = [[number, number], [number, number]];

// ── Vector operations ──────────────────────────────────────────────────────────

export function vecAdd(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]];
}

export function vecScale(v: Vec2, s: number): Vec2 {
  return [v[0] * s, v[1] * s];
}

export function vecLength(v: Vec2): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}

export function vecNormalize(v: Vec2): Vec2 {
  const len = vecLength(v);
  if (len === 0) return [0, 0];
  return [v[0] / len, v[1] / len];
}

export function vecDot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

export function vecAngle(a: Vec2, b: Vec2): number {
  const dot = vecDot(a, b);
  const lenA = vecLength(a);
  const lenB = vecLength(b);
  if (lenA === 0 || lenB === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (lenA * lenB))));
}

/** Project vector a onto vector b */
export function vecProject(a: Vec2, b: Vec2): Vec2 {
  const lenBSq = vecDot(b, b);
  if (lenBSq === 0) return [0, 0];
  const scalar = vecDot(a, b) / lenBSq;
  return vecScale(b, scalar);
}

// ── Matrix operations ──────────────────────────────────────────────────────────

export function matMulVec(m: Mat2, v: Vec2): Vec2 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1],
    m[1][0] * v[0] + m[1][1] * v[1],
  ];
}

export function matDeterminant(m: Mat2): number {
  return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

export function matTranspose(m: Mat2): Mat2 {
  return [
    [m[0][0], m[1][0]],
    [m[0][1], m[1][1]],
  ];
}

/** Interpolate between two matrices for animation (t=0 → identity, t=1 → m) */
export function matLerp(a: Mat2, b: Mat2, t: number): Mat2 {
  return [
    [a[0][0] + (b[0][0] - a[0][0]) * t, a[0][1] + (b[0][1] - a[0][1]) * t],
    [a[1][0] + (b[1][0] - a[1][0]) * t, a[1][1] + (b[1][1] - a[1][1]) * t],
  ];
}

export const MAT_IDENTITY: Mat2 = [[1, 0], [0, 1]];

// ── Eigenvalue / Eigenvector for 2×2 real matrices ────────────────────────────

export interface EigenResult {
  lambda1: number;
  lambda2: number;
  vec1: Vec2;
  vec2: Vec2;
  isReal: boolean;
}

export function eigen2x2(m: Mat2): EigenResult {
  const a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
  const trace = a + d;
  const det = a * d - b * c;
  const discriminant = trace * trace - 4 * det;

  if (discriminant < 0) {
    // Complex eigenvalues — return degenerate result
    return {
      lambda1: trace / 2,
      lambda2: trace / 2,
      vec1: [1, 0],
      vec2: [0, 1],
      isReal: false,
    };
  }

  const sqrtDisc = Math.sqrt(discriminant);
  const lambda1 = (trace + sqrtDisc) / 2;
  const lambda2 = (trace - sqrtDisc) / 2;

  const eigenvector = (lambda: number): Vec2 => {
    // Solve (A - λI)v = 0
    const row0x = a - lambda;
    const row0y = b;
    const row1x = c;
    const row1y = d - lambda;

    // Pick the row with larger magnitude to avoid dividing by zero
    if (Math.abs(row0x) + Math.abs(row0y) >= Math.abs(row1x) + Math.abs(row1y)) {
      if (Math.abs(row0x) > 1e-10 || Math.abs(row0y) > 1e-10) {
        const v: Vec2 = [-row0y, row0x];
        return vecNormalize(v);
      }
    } else {
      if (Math.abs(row1x) > 1e-10 || Math.abs(row1y) > 1e-10) {
        const v: Vec2 = [-row1y, row1x];
        return vecNormalize(v);
      }
    }
    return [1, 0];
  };

  return {
    lambda1,
    lambda2,
    vec1: eigenvector(lambda1),
    vec2: eigenvector(lambda2),
    isReal: true,
  };
}

/** Classify a 2×2 transformation */
export function classifyTransform(m: Mat2): string {
  const det = matDeterminant(m);
  const a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
  const isOrthogonal = Math.abs(a * a + c * c - 1) < 0.01 &&
                       Math.abs(b * b + d * d - 1) < 0.01 &&
                       Math.abs(a * b + c * d) < 0.01;
  const isIdentity = Math.abs(a - 1) < 0.01 && Math.abs(d - 1) < 0.01 &&
                     Math.abs(b) < 0.01 && Math.abs(c) < 0.01;

  if (isIdentity) return "Identity";
  if (Math.abs(det) < 0.001) return "Singular (collapses space)";
  if (isOrthogonal && det > 0) return "Rotation";
  if (isOrthogonal && det < 0) return "Reflection";
  if (Math.abs(b) < 0.01 && Math.abs(c) < 0.01) {
    if (Math.abs(a - d) < 0.01) return `Uniform Scale (×${a.toFixed(2)})`;
    return "Non-uniform Scale";
  }
  if (Math.abs(c) < 0.01 && Math.abs(b) > 0.1) return "Horizontal Shear";
  if (Math.abs(b) < 0.01 && Math.abs(c) > 0.1) return "Vertical Shear";
  return "Linear Transformation";
}

// ── Preset matrices ───────────────────────────────────────────────────────────

export const MATRIX_PRESETS: Record<string, { label: string; matrix: Mat2 }> = {
  identity:  { label: "Identity",   matrix: [[1, 0], [0, 1]] },
  rotate90:  { label: "Rotate 90°", matrix: [[0, -1], [1, 0]] },
  shear:     { label: "Shear",      matrix: [[1, 0.8], [0, 1]] },
  reflectX:  { label: "Reflect X",  matrix: [[1, 0], [0, -1]] },
  scale2x:   { label: "Scale 2×",   matrix: [[2, 0], [0, 2]] },
};

export const EIGEN_PRESETS: Record<string, { label: string; matrix: Mat2 }> = {
  sym:       { label: "[[2,1],[1,2]]", matrix: [[2, 1], [1, 2]] },
  diag:      { label: "[[3,0],[0,1]]", matrix: [[3, 0], [0, 1]] },
  scale:     { label: "[[2,0],[0,2]]", matrix: [[2, 0], [0, 2]] },
  rotate:    { label: "Rotate 45°",   matrix: [[0.707, -0.707], [0.707, 0.707]] },
};

// ── Coordinate helpers ────────────────────────────────────────────────────────

/** Convert math coords → SVG pixel coords */
export function toSVG(
  v: Vec2,
  cx: number,
  cy: number,
  scale: number,
): Vec2 {
  return [cx + v[0] * scale, cy - v[1] * scale];
}

/** Clamp a value */
export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/** Format a number to a fixed number of significant decimals, stripping trailing zeros */
export function fmt(n: number, decimals = 2): string {
  return parseFloat(n.toFixed(decimals)).toString();
}

/** Degrees ↔ radians */
export const DEG = Math.PI / 180;
export const RAD = 180 / Math.PI;
