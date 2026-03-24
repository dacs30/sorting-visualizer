"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/sorting", label: "Sorting" },
  { href: "/linear-algebra", label: "Linear Algebra" },
  { href: "/machine-learning", label: "Machine Learning" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header
      style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}
      className="sticky top-0 z-50"
    >
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center gap-8">
        <Link
          href="/"
          className="font-serif text-lg font-semibold tracking-tight"
          style={{ color: "var(--primary)" }}
        >
          AlgoHub
        </Link>
        <nav className="flex items-center gap-1">
          {links.map(({ href, label }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className="px-3 py-1.5 rounded text-sm transition-colors"
                style={{
                  color: active ? "var(--primary)" : "var(--muted)",
                  background: active ? "var(--surface2)" : "transparent",
                  fontWeight: active ? 600 : 400,
                }}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
