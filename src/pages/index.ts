export async function render(app: HTMLElement) {
    document.title = "Edge ML Demo";

    app.innerHTML = `
    <div class="min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-300 selection:text-slate-950">
        <div class="mx-auto max-w-6xl px-5 py-8 md:px-8 md:py-10">
            <header class="rounded-3xl border border-slate-800 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.22),_transparent_45%),radial-gradient(circle_at_bottom_right,_rgba(251,191,36,0.18),_transparent_45%),rgba(15,23,42,0.8)] p-6 md:p-8">
                <div class="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                    <div class="space-y-3">
                        <span class="inline-flex rounded-full border border-emerald-300/40 bg-emerald-300/10 px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.18em] text-emerald-200">Edge AI Playground</span>
                        <h1 class="text-3xl font-black tracking-tight text-white md:text-5xl">Local Inference with WASM and WebGPU</h1>
                        <p class="max-w-3xl text-sm text-slate-300 md:text-base">Demo catalog showing that ML can run in the browser with low latency and no sensitive data sent to the cloud.</p>
                    </div>
                    <div class="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-4 font-mono text-xs text-slate-300">
                        <div class="flex items-center justify-between gap-4"><span>Demo status</span><span class="text-emerald-300">ONLINE</span></div>
                        <div class="mt-2 text-[11px] text-slate-400">5 demos ready · 1 in development</div>
                    </div>
                </div>
            </header>

            <section class="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                <a href="./evaluation" class="group rounded-2xl border border-emerald-600/40 bg-emerald-950/20 p-5 transition hover:-translate-y-1 hover:border-emerald-400">
                    <div class="mb-4 flex items-center justify-between">
                        <span class="text-xs font-bold uppercase tracking-[0.16em] text-emerald-300">Available</span>
                        <span class="rounded-full bg-emerald-400/20 px-2 py-0.5 text-[10px] font-mono text-emerald-200">BERT</span>
                    </div>
                    <h2 class="text-xl font-extrabold text-white">Sentiment Analysis</h2>
                    <p class="mt-3 text-sm text-slate-300">Multilingual text classification with real-time load and inference telemetry.</p>
                </a>

                <a href="./vlm" class="group rounded-2xl border border-emerald-600/40 bg-emerald-950/20 p-5 transition hover:-translate-y-1 hover:border-emerald-400">
                    <div class="mb-4 flex items-center justify-between">
                        <span class="text-xs font-bold uppercase tracking-[0.16em] text-emerald-300">Available</span>
                        <span class="rounded-full bg-emerald-400/20 px-2 py-0.5 text-[10px] font-mono text-emerald-200">Florence-2</span>
                    </div>
                    <h2 class="text-xl font-extrabold text-white">VLM Master Scan</h2>
                    <p class="mt-3 text-sm text-slate-300">Description, OCR, detection, and open-vocabulary inspection with direct grounding on the image.</p>
                </a>

                <a href="./llm" class="group rounded-2xl border border-emerald-600/40 bg-emerald-950/20 p-5 transition hover:-translate-y-1 hover:border-emerald-400">
                    <div class="mb-4 flex items-center justify-between">
                        <span class="text-xs font-bold uppercase tracking-[0.16em] text-emerald-300">Available</span>
                        <span class="rounded-full bg-emerald-400/20 px-2 py-0.5 text-[10px] font-mono text-emerald-200">SmolLM2/Qwen2.5</span>
                    </div>
                    <h2 class="text-xl font-extrabold text-white">LLM Chat Local</h2>
                    <p class="mt-3 text-sm text-slate-300">Context-aware chat with inference timing and fully local execution in the browser.</p>
                </a>

                <a href="./semantic" class="group rounded-2xl border border-emerald-600/40 bg-emerald-950/20 p-5 transition hover:-translate-y-1 hover:border-emerald-400">
                    <div class="mb-4 flex items-center justify-between">
                        <span class="text-xs font-bold uppercase tracking-[0.16em] text-emerald-300">Available</span>
                        <span class="rounded-full bg-emerald-400/20 px-2 py-0.5 text-[10px] font-mono text-emerald-200">MiniLM</span>
                    </div>
                    <h2 class="text-xl font-extrabold text-slate-100">Semantic Search</h2>
                    <p class="mt-3 text-sm text-slate-300">Semantic text ranking with local embeddings and cosine similarity.</p>
                </a>

                <a href="./rag" class="group rounded-2xl border border-emerald-600/40 bg-emerald-950/20 p-5 transition hover:-translate-y-1 hover:border-emerald-400">
                    <div class="mb-4 flex items-center justify-between">
                        <span class="text-xs font-bold uppercase tracking-[0.16em] text-emerald-300">Available</span>
                        <span class="rounded-full bg-emerald-400/20 px-2 py-0.5 text-[10px] font-mono text-emerald-200">RAG</span>
                    </div>
                    <h2 class="text-xl font-extrabold text-slate-100">Retrieval-Augmented Generation</h2>
                    <p class="mt-3 text-sm text-slate-300">Local semantic retrieval plus grounded generation in one flow.</p>
                </a>

                <article class="rounded-2xl border border-slate-700 bg-slate-900/60 p-5 opacity-85">
                    <div class="mb-4 flex items-center justify-between">
                        <span class="text-xs font-bold uppercase tracking-[0.16em] text-slate-300">Coming soon</span>
                        <span class="rounded-full bg-slate-700/60 px-2 py-0.5 text-[10px] font-mono text-slate-300">YOLOv8</span>
                    </div>
                    <h2 class="text-xl font-extrabold text-slate-100">Object Detection</h2>
                    <p class="mt-3 text-sm text-slate-400">Video/image detection for visual inspection and alerts on edge devices.</p>
                </article>
            </section>
        </div>
    </div>
    `;
}
