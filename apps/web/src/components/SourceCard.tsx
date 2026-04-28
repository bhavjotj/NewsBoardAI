import { ExternalLink } from "lucide-react";

import type { SourceCardData } from "../types/dashboard";

interface SourceCardProps {
  source: SourceCardData;
}

export function SourceCard({ source }: SourceCardProps) {
  const publishedDate = formatDate(source.published_at);

  return (
    <article className="rounded-lg border border-line bg-card p-3">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer"
            className="line-clamp-2 text-sm font-semibold leading-5 text-neutral-100 transition hover:text-cyan-200"
          >
            {source.title}
          </a>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-neutral-500">
            <span className="text-neutral-300">{source.source}</span>
            {publishedDate && <span>{publishedDate}</span>}
          </div>
        </div>
        <a
          href={source.url}
          target="_blank"
          rel="noreferrer"
          aria-label={`Open ${source.title}`}
          className="rounded-md border border-line p-1.5 text-neutral-400 transition hover:border-cyan-400 hover:text-cyan-300"
        >
          <ExternalLink className="h-4 w-4" aria-hidden="true" />
        </a>
      </div>
      {source.snippet && (
        <p className="line-clamp-2 mt-2 text-xs leading-5 text-neutral-400">
          {source.snippet}
        </p>
      )}
    </article>
  );
}

function formatDate(value: string | null) {
  if (!value) {
    return "";
  }

  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
