import { Sparkles } from "lucide-react";

const EXAMPLES = [
  "Tesla",
  "Netflix",
  "Nintendo Switch 2",
  "Toronto Raptors",
  "Bitcoin",
];

interface EmptyStateProps {
  onExampleSearch: (query: string) => void;
}

export function EmptyState({ onExampleSearch }: EmptyStateProps) {
  return (
    <section className="rounded-lg border border-dashed border-line bg-card/70 p-5 text-center">
      <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
        <Sparkles className="h-5 w-5" aria-hidden="true" />
      </div>
      <h2 className="text-sm font-semibold text-neutral-100">
        Search any live topic
      </h2>
      <p className="mt-2 text-sm leading-5 text-neutral-400">
        Get a compact signal, key tags, possible impact, and source links.
      </p>
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onExampleSearch(example)}
            className="rounded-full border border-line bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition hover:border-cyan-400 hover:text-cyan-200"
          >
            {example}
          </button>
        ))}
      </div>
    </section>
  );
}
