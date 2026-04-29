import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  base: mode === "extension" ? "./" : "/",
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
}));
