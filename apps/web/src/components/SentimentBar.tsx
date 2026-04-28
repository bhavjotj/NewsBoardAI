import type { SentimentSummary } from "../types/dashboard";

interface SentimentBarProps {
  sentiment: SentimentSummary;
}

export function SentimentBar({ sentiment }: SentimentBarProps) {
  const clampedScore = Math.max(-1, Math.min(1, sentiment.score));
  const position = `${((clampedScore + 1) / 2) * 100}%`;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="font-medium capitalize text-neutral-100">
          {sentiment.label}
        </span>
        <span className="tabular-nums text-neutral-400">
          {clampedScore.toFixed(2)}
        </span>
      </div>
      <div className="relative h-2 rounded-full bg-gradient-to-r from-rose-500 via-neutral-600 to-emerald-400">
        <span
          className="absolute top-1/2 h-4 w-1.5 -translate-y-1/2 rounded-full bg-white shadow"
          style={{ left: position }}
          aria-hidden="true"
        />
      </div>
      <div className="mt-1 flex justify-between text-[10px] uppercase tracking-wide text-neutral-500">
        <span>Negative</span>
        <span>Positive</span>
      </div>
    </div>
  );
}
