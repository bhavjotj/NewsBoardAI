import { AlertCircle } from "lucide-react";
import { useState } from "react";

import { fetchDashboard } from "./api/news";
import { DashboardView } from "./components/DashboardView";
import { EmptyState } from "./components/EmptyState";
import { SearchPanel } from "./components/SearchPanel";
import type { DashboardResponse } from "./types/dashboard";
import { formatUpdatedAt } from "./utils/format";

const LOADING_STEPS = [
  "Fetching recent news",
  "Analyzing signal",
  "Building dashboard",
];

function App() {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  async function handleSearch(query: string, maxResults: number) {
    setIsLoading(true);
    setError("");
    setLoadingStep(0);

    const stepTimer = window.setInterval(() => {
      setLoadingStep((currentStep) =>
        Math.min(currentStep + 1, LOADING_STEPS.length - 1),
      );
    }, 900);

    try {
      const result = await fetchDashboard({
        query,
        max_results: maxResults,
        use_real_news: true,
        use_ml: true,
      });
      setDashboard(result);
      setLastUpdated(new Date());
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Something went wrong while fetching news.",
      );
    } finally {
      window.clearInterval(stepTimer);
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 px-3 py-4 text-neutral-100">
      <div className="mx-auto flex w-full max-w-[460px] flex-col gap-3">
        <SearchPanel
          isLoading={isLoading}
          loadingLabel={LOADING_STEPS[loadingStep]}
          onSearch={handleSearch}
        />

        {error && (
          <section className="flex gap-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-100">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <div>
              <p className="font-semibold">Backend unavailable</p>
              <p className="mt-1 text-rose-100/80">{error}</p>
            </div>
          </section>
        )}

        {isLoading && (
          <section className="rounded-lg border border-line bg-card p-4">
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-neutral-100">
                {LOADING_STEPS[loadingStep]}
              </p>
              <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-300" />
            </div>
            <div className="mt-4 h-20 animate-pulse rounded bg-neutral-900" />
            <div className="mt-3 h-16 animate-pulse rounded bg-neutral-900" />
          </section>
        )}

        {!isLoading && !dashboard && !error && (
          <EmptyState onExampleSearch={(query) => handleSearch(query, 3)} />
        )}
        {!isLoading && dashboard && (
          <DashboardView
            dashboard={dashboard}
            lastUpdated={
              lastUpdated ? `Last updated ${formatUpdatedAt(lastUpdated)}` : ""
            }
          />
        )}
      </div>
    </main>
  );
}

export default App;
