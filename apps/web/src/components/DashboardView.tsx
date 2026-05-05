import { Activity, AlertCircle, Sparkles, TrendingUp } from "lucide-react";

import type { DashboardResponse } from "../types/dashboard";
import { formatLabel, formatTopic } from "../utils/format";
import { Badge } from "./Badge";
import { SentimentBar } from "./SentimentBar";
import { SourceCard } from "./SourceCard";

interface DashboardViewProps {
  dashboard: DashboardResponse;
  lastUpdated: string;
}

export function DashboardView({ dashboard, lastUpdated }: DashboardViewProps) {
  const InsightIcon = insightIcon(dashboard.overall_signal);
  const tone = signalTone(dashboard.overall_signal);

  return (
    <div className="space-y-3">
      <section className="rounded-lg border border-line bg-card p-4 shadow-soft">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[11px] font-medium uppercase tracking-wide text-neutral-500">
              {dashboard.time_window}
            </p>
            <h1 className="mt-1 truncate text-xl font-semibold text-neutral-50">
              {formatTopic(dashboard.topic)}
            </h1>
            {lastUpdated && (
              <p className="mt-1 text-xs text-neutral-500">{lastUpdated}</p>
            )}
          </div>
          <Activity className="mt-1 h-5 w-5 shrink-0 text-cyan-300" />
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <Badge tone="info">{formatLabel(dashboard.data_source)}</Badge>
          {dashboard.analysis_source && (
            <Badge tone="info">{formatLabel(dashboard.analysis_source)}</Badge>
          )}
          {usesLocalLlm(dashboard) && (
            <Badge tone="info">Local LLM</Badge>
          )}
          {dashboard.torch_used && <Badge tone="info">neural topic</Badge>}
          <Badge>{dashboard.detected_mode}</Badge>
          <Badge tone={tone}>{dashboard.overall_signal}</Badge>
          <Badge tone={confidenceTone(dashboard.confidence)}>
            {dashboard.confidence} confidence
          </Badge>
        </div>
      </section>

      <section className="rounded-lg border border-line bg-card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-neutral-100">Signal</h2>
          <TrendingUp className="h-4 w-4 text-emerald-300" aria-hidden="true" />
        </div>
        <SentimentBar sentiment={dashboard.sentiment} />
        <div className="mt-3 flex flex-wrap gap-2">
          {dashboard.event_tags.map((tag) => (
            <Badge key={tag}>{tag}</Badge>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-line bg-card p-4">
        <h2 className="mb-2 text-sm font-semibold text-neutral-100">Brief</h2>
        <p className="text-sm leading-6 text-neutral-300">{dashboard.brief}</p>
      </section>

      <section className={`rounded-lg border p-4 ${impactCardClass(tone)}`}>
        <div className="mb-2 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-neutral-100">
            <InsightIcon className="h-4 w-4" aria-hidden="true" />
            <h2 className="text-sm font-semibold">Possible impact</h2>
          </div>
          <Badge tone={tone}>impact</Badge>
        </div>
        <p className="text-sm leading-6 text-neutral-200">
          {dashboard.possible_impact}
        </p>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-neutral-100">Sources</h2>
          <span className="text-xs text-neutral-500">
            {dashboard.sources.length} shown
          </span>
        </div>
        <div className="space-y-2">
          {dashboard.sources.map((source) => (
            <SourceCard key={source.url} source={source} />
          ))}
        </div>
      </section>
    </div>
  );
}

function signalTone(signal: string) {
  if (signal === "positive") {
    return "good";
  }
  if (signal === "negative") {
    return "bad";
  }
  if (signal === "mixed" || signal === "unclear") {
    return "warn";
  }
  return "neutral";
}

function confidenceTone(confidence: string) {
  if (confidence === "high") {
    return "good";
  }
  if (confidence === "low") {
    return "warn";
  }
  return "info";
}

function impactCardClass(tone: ReturnType<typeof signalTone>) {
  if (tone === "good") {
    return "border-emerald-500/25 bg-emerald-500/10";
  }
  if (tone === "bad") {
    return "border-rose-500/25 bg-rose-500/10";
  }
  if (tone === "warn") {
    return "border-amber-500/25 bg-amber-500/10";
  }
  return "border-line bg-card";
}

function insightIcon(signal: string) {
  if (signal === "positive") {
    return Sparkles;
  }
  if (signal === "negative" || signal === "mixed") {
    return AlertCircle;
  }
  return Activity;
}

function usesLocalLlm(dashboard: DashboardResponse) {
  return [dashboard.brief_source, dashboard.possible_impact_source].some((source) =>
    source === "ollama" || source === "ollama_partial"
  );
}
