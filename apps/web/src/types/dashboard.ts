export type DataSource = "mock" | "google_news_rss" | "fallback_mock";
export type AnalysisSource = "rule_based" | "hybrid_ml" | "hybrid_ml_fallback";

export type DashboardMode =
  | "general"
  | "business"
  | "sports"
  | "gaming"
  | "politics";

export type SentimentLabel = "positive" | "neutral" | "negative" | "mixed";

export interface SentimentSummary {
  label: SentimentLabel;
  score: number;
}

export interface SourceCardData {
  title: string;
  source: string;
  published_at: string | null;
  snippet: string;
  url: string;
}

export interface DashboardResponse {
  topic: string;
  data_source: DataSource;
  analysis_source?: AnalysisSource;
  detected_mode: DashboardMode;
  time_window: string;
  overall_signal: string;
  brief: string;
  sentiment: SentimentSummary;
  event_tags: string[];
  sources: SourceCardData[];
  confidence: "low" | "medium" | "high";
  possible_impact: string;
  torch_used?: boolean;
  torch_available?: boolean;
}

export interface DashboardRequest {
  query: string;
  max_results: number;
  use_real_news: true;
  use_ml: true;
  use_torch: true;
}
