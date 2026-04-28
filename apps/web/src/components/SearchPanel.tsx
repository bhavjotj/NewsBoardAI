import { Search } from "lucide-react";
import { FormEvent, useState } from "react";

interface SearchPanelProps {
  isLoading: boolean;
  loadingLabel: string;
  onSearch: (query: string, maxResults: number) => void;
}

export function SearchPanel({
  isLoading,
  loadingLabel,
  onSearch,
}: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const [maxResults, setMaxResults] = useState(3);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const cleanedQuery = query.trim();
    if (!cleanedQuery || isLoading) {
      return;
    }
    onSearch(cleanedQuery, maxResults);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-lg border border-line bg-card p-3 shadow-soft"
    >
      <div className="mb-3 flex items-center gap-2 text-neutral-300">
        <Search className="h-4 w-4 text-cyan-300" aria-hidden="true" />
        <span className="text-sm font-semibold">NewsBoardAI</span>
      </div>

      <div className="flex gap-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search Tesla, Nintendo Switch 2..."
          className="min-w-0 flex-1 rounded-md border border-line bg-neutral-950 px-3 py-2 text-sm text-neutral-100 outline-none transition placeholder:text-neutral-600 focus:border-cyan-400"
        />
        <select
          value={maxResults}
          onChange={(event) => setMaxResults(Number(event.target.value))}
          className="w-16 rounded-md border border-line bg-neutral-950 px-2 py-2 text-sm text-neutral-200 outline-none focus:border-cyan-400"
          aria-label="Max results"
        >
          {[2, 3, 4, 5].map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </div>

      <button
        type="submit"
        disabled={isLoading || !query.trim()}
        className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-md bg-cyan-300 px-3 py-2 text-sm font-semibold text-neutral-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-400"
      >
        {isLoading ? loadingLabel : "Build dashboard"}
      </button>
    </form>
  );
}
