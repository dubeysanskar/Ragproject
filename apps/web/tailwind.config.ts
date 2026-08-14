import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        greenDeep: "#0A6B3C",
        greenDark: "#064D2B",
        yellow: "#FFD400",
        pink: "#EC1E79",
        cream: "#FBF7EC",
        ink: "#111111",
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
        deva: ["var(--font-deva)", "serif"],
      },
    },
  },
  plugins: [],
};

export default config;
