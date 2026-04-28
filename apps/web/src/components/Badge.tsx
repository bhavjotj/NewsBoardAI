type BadgeTone = "neutral" | "good" | "warn" | "bad" | "info";

interface BadgeProps {
  children: React.ReactNode;
  tone?: BadgeTone;
}

const toneClass: Record<BadgeTone, string> = {
  neutral: "border-line bg-neutral-900 text-neutral-300",
  good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  warn: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  bad: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  info: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
};

export function Badge({ children, tone = "neutral" }: BadgeProps) {
  return (
    <span
      className={`inline-flex max-w-full items-center rounded-full border px-2.5 py-1 text-[11px] font-medium capitalize leading-none ${toneClass[tone]}`}
    >
      {children}
    </span>
  );
}
