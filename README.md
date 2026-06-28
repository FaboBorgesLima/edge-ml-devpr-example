# DevPR Edge ML Demo

Interactive browser demos for local AI inference with WebGPU and WASM using open-source models and Transformers.js.

https://faboborgeslima.github.io/edge-ml-devpr-example/

# IMPORTANT

The models are extremely large and heavy. This is only a proof-of-concept demo. For production consider using smaller models or server-side inference (or a mixture of both, like using LLMs as routers and RAGs with tons of retrieval data).

## Features

- Sentiment analysis (BERT)
- VLM master scan (Florence-2)
- Local LLM chat (SmolLM2 / Qwen2.5)
- Semantic search (MiniLM)
- RAG (retrieval + generation)
- Object detection with YOLOv8n (Nano, lightest YOLOv8 variant)

## Tech Stack

- Vite + TypeScript
- Tailwind CSS v4
- Transformers.js (`@huggingface/transformers`)

## Getting Started

### 1. Install dependencies

```bash
npm install
```

### 2. Start development server

```bash
npm run dev
```

Open the local URL shown in the terminal.

### 3. Production build

```bash
npm run build
npm run preview
```

## Deploy (GitHub Pages)

```bash
npm run deploy
```

## Notes

- First load can be slower because models are downloaded and cached.
- WebGPU performance depends on browser and GPU support.
- WASM mode has broader compatibility but lower performance.
