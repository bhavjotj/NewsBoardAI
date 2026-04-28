/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        panel: "#111111",
        card: "#1a1a1a",
        line: "#2b2b2b",
      },
      boxShadow: {
        soft: "0 18px 45px rgba(0, 0, 0, 0.28)",
      },
    },
  },
  plugins: [],
};
