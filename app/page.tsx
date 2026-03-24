import Link from "next/link";
import HeroAnimation from "./components/HeroAnimation";

const topics = [
  {
    href: "/sorting",
    label: "Sorting Algorithms",
    description:
      "Watch bubble, merge, quick, and heap sort work through data in real time. Adjust speed, compare approaches, and build intuition for algorithmic complexity.",
    tags: ["O(n log n)", "Comparisons", "In-place"],
    available: true,
  },
  {
    href: "/linear-algebra",
    label: "Linear Algebra",
    description:
      "Visualize vectors, matrix transformations, dot products, and eigenvalues in 2D. See the geometry behind the math.",
    tags: ["Vectors", "Matrices", "Transformations"],
    available: true,
  },
  {
    href: "/machine-learning",
    label: "Machine Learning",
    description:
      "Explore gradient descent, k-means clustering, neural network forward passes, and decision boundaries — all animated and interactive.",
    tags: ["Gradient Descent", "Clustering", "Neural Nets"],
    available: true,
  },
];

export default function Home() {
  return (
    <main className="flex-1">
      {/* Hero */}
      <section className="dot-pattern border-b" style={{ borderColor: "var(--border)" }}>
        <div className="max-w-6xl mx-auto px-6 py-20 md:py-28 flex flex-col md:flex-row items-center gap-12">
          {/* Left: text */}
          <div className="flex-1 min-w-0">
            <p
              className="text-sm font-mono tracking-widest uppercase mb-4"
              style={{ color: "var(--primary)" }}
            >
              Interactive Learning
            </p>
            <h1
              className="font-serif text-4xl md:text-6xl font-bold leading-tight mb-6"
              style={{ color: "var(--fg)" }}
            >
              Algorithms &amp; Math,
              <br />
              <span style={{ color: "var(--primary)" }}>made visual.</span>
            </h1>
            <p className="text-lg max-w-xl" style={{ color: "var(--muted)" }}>
              Explore sorting algorithms, linear algebra, and machine learning through
              interactive animations. No formulas without intuition.
            </p>
          </div>

          {/* Right: live animation */}
          <div className="w-full md:w-80 shrink-0">
            <HeroAnimation />
          </div>
        </div>
      </section>

      {/* Topic cards */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2
          className="text-xs font-mono tracking-widest uppercase mb-8"
          style={{ color: "var(--muted)" }}
        >
          Topics
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {topics.map(({ href, label, description, tags, available }) => (
            <TopicCard
              key={href}
              href={href}
              label={label}
              description={description}
              tags={tags}
              available={available}
            />
          ))}
        </div>
      </section>

      <footer
        className="border-t py-6 text-center text-xs font-mono"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}
      >
        © {new Date().getFullYear()} Danilo Correia. All rights reserved.
      </footer>
    </main>
  );
}

function TopicCard({
  href,
  label,
  description,
  tags,
  available,
}: {
  href: string;
  label: string;
  description: string;
  tags: string[];
  available: boolean;
}) {
  const card = (
    <div
      className="h-full rounded-xl border p-6 flex flex-col gap-4 transition-shadow"
      style={{
        background: "var(--surface)",
        borderColor: "var(--border)",
        opacity: available ? 1 : 0.6,
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-serif text-xl font-semibold" style={{ color: "var(--fg)" }}>
          {label}
        </h3>
        {!available && (
          <span
            className="text-xs font-mono px-2 py-0.5 rounded-full shrink-0"
            style={{ background: "var(--surface2)", color: "var(--muted)" }}
          >
            soon
          </span>
        )}
      </div>
      <p className="text-sm leading-relaxed flex-1" style={{ color: "var(--muted)" }}>
        {description}
      </p>
      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span
            key={tag}
            className="text-xs font-mono px-2 py-0.5 rounded"
            style={{ background: "var(--surface2)", color: "var(--primary)" }}
          >
            {tag}
          </span>
        ))}
      </div>
      {available && (
        <span
          className="text-sm font-medium mt-1"
          style={{ color: "var(--primary)" }}
        >
          Explore →
        </span>
      )}
    </div>
  );

  return available ? (
    <Link href={href} className="block group hover:shadow-md rounded-xl transition-shadow">
      {card}
    </Link>
  ) : (
    <div className="block cursor-default">{card}</div>
  );
}
