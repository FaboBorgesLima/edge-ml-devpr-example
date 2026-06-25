# DevPR Edge ML Demo

Interactive browser demos for local AI inference with WebGPU and WASM.

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

## YOLO Page

The YOLO demo is available at `/yolo` and uses:

- Model: `onnx-community/yolov8n`
- Task: object detection
- Runtime: WebGPU when available, WASM fallback otherwise

You can upload an image, set confidence threshold, and inspect detected objects with bounding boxes and confidence values.

## Notes

- First load can be slower because models are downloaded and cached.
- WebGPU performance depends on browser and GPU support.
- WASM mode has broader compatibility but lower performance.
