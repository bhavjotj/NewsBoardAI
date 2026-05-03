const SMALL_WORDS = new Set(["and", "or", "the", "of", "for", "to", "in"]);
const KNOWN_WORDS: Record<string, string> = {
  ai: "AI",
  bitcoin: "Bitcoin",
  microsoft: "Microsoft",
  netflix: "Netflix",
  nintendo: "Nintendo",
  raptors: "Raptors",
  switch: "Switch",
  tesla: "Tesla",
  toronto: "Toronto",
};

export function formatTopic(value: string) {
  // Search text becomes a dashboard title, so preserve known product casing.
  return value
    .trim()
    .split(/\s+/)
    .map((word, index) => formatTopicWord(word, index))
    .join(" ");
}

export function formatLabel(value: string) {
  return value.replace(/_/g, " ");
}

export function formatUpdatedAt(value: Date) {
  return new Intl.DateTimeFormat("en", {
    hour: "numeric",
    minute: "2-digit",
  }).format(value);
}

function formatTopicWord(word: string, index: number) {
  if (/^\d+$/.test(word)) {
    return word;
  }

  const lowerWord = word.toLowerCase();
  if (KNOWN_WORDS[lowerWord]) {
    return KNOWN_WORDS[lowerWord];
  }
  if (index > 0 && SMALL_WORDS.has(lowerWord)) {
    return lowerWord;
  }

  return lowerWord.charAt(0).toUpperCase() + lowerWord.slice(1);
}
