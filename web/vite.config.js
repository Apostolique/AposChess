import { defineConfig } from 'vite'

// Relative base so the built site works under the GitHub Pages project path
// (https://apostolique.github.io/AposChess/) without hard-coding the repo name.
export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
    target: 'es2020',
  },
})
