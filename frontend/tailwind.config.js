/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        starbucks: {
          green: "#036635",
          mint: "#a4cbb4",
          light: "#d4e9e2",
          cream: "#f8f8f1",
          brown: "#4b2e2b",
          highlight: "#94bfa2",
        },
      },
    },
  },
  plugins: [],
}

