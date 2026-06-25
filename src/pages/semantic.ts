import { hasGPU } from "../lib/has-gpu";
import { Timer } from "../lib/timer";
import {
    rankBySemanticSimilarity,
    warmupSemanticModel,
} from "../services/semantic-service";

const SAMPLE_DOCS = [
    "WebGPU can accelerate embedding and inference workloads in supported browsers.",
    "WASM is broadly compatible and useful fallback when WebGPU is unavailable.",
    "Semantic search ranks passages by meaning using embedding vectors and cosine similarity.",
    "For private data, local embeddings avoid sending text to cloud APIs.",
    "Use chunking to split long documents before indexing for retrieval.",
    "A reranker can improve top results after initial vector retrieval.",
].join("\n");

export async function render(app: HTMLElement) {
    document.title = "Semantic Search";

    app.innerHTML = `
    <div class="min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-300 selection:text-slate-950">
      <div class="mx-auto max-w-5xl px-4 py-6 md:px-8 md:py-10">
        <header class="mb-5 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 md:p-6">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="flex items-center gap-2">
                <span class="rounded-full border border-emerald-300/40 bg-emerald-300/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-200">Edge Search</span>
                <a href="${import.meta.env.BASE_URL}" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">back to catalog</a>
              </div>
              <h1 class="mt-2 text-2xl font-black tracking-tight text-white md:text-4xl">Semantic Search (Local)</h1>
              <p class="mt-2 max-w-3xl text-sm text-slate-300">Rank text snippets by meaning using local embeddings in the browser.</p>
            </div>
            <div class="rounded-xl border border-slate-700/70 bg-slate-950/60 p-3 text-xs font-mono">
              <div class="flex items-center justify-between gap-4"><span class="text-slate-400">Runtime</span><span id="hardware-badge" class="text-emerald-300">Detecting...</span></div>
              <div class="mt-2 flex items-center justify-between gap-4"><span class="text-slate-500">Model load</span><span id="load-ms" class="font-bold text-emerald-300">-- ms</span></div>
              <div class="mt-1 flex items-center justify-between gap-4"><span class="text-slate-500">Search</span><span id="search-ms" class="font-bold text-emerald-300">-- ms</span></div>
            </div>
          </div>
        </header>

        <main class="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_300px]">
          <section class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <label for="docs-input" class="text-xs font-bold uppercase tracking-wider text-slate-300">Corpus (one snippet per line)</label>
            <textarea id="docs-input" rows="8" class="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-emerald-300 focus:outline-none">${SAMPLE_DOCS}</textarea>

            <label for="query-input" class="mt-4 block text-xs font-bold uppercase tracking-wider text-slate-300">Query</label>
            <div class="mt-2 flex gap-2">
              <input id="query-input" type="text" class="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-emerald-300 focus:outline-none" placeholder="Example: browser fallback when webgpu is unavailable" />
              <button id="search-btn" class="rounded-lg bg-emerald-300 px-4 py-2 text-xs font-black uppercase tracking-wider text-slate-950">Search</button>
            </div>

            <div id="results" class="mt-4 space-y-2"></div>
          </section>

          <aside class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <h2 class="text-sm font-black uppercase tracking-wider text-white">Tips</h2>
            <ul class="mt-3 list-disc space-y-2 pl-4 text-xs text-slate-300">
              <li>Use short snippets for better ranking.</li>
              <li>Keep one topic per line.</li>
              <li>Use specific queries, not generic words.</li>
            </ul>
          </aside>
        </main>
      </div>
    </div>
    `;

    await boot();
}

async function boot() {
    const hardwareBadge = document.getElementById("hardware-badge") as HTMLSpanElement;
    const loadMs = document.getElementById("load-ms") as HTMLSpanElement;
    const searchMs = document.getElementById("search-ms") as HTMLSpanElement;
    const docsInput = document.getElementById("docs-input") as HTMLTextAreaElement;
    const queryInput = document.getElementById("query-input") as HTMLInputElement;
    const searchBtn = document.getElementById("search-btn") as HTMLButtonElement;
    const results = document.getElementById("results") as HTMLDivElement;

    const useGpu = await hasGPU();
    hardwareBadge.innerText = useGpu ? "WebGPU" : "WASM";

    const [, modelLoadMs] = await Timer.wrap(() => warmupSemanticModel(useGpu))();
    loadMs.innerText = `${modelLoadMs} ms`;

    const runSearch = async () => {
      const query = queryInput.value.trim();
      const docs = docsInput.value
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      if (!query || docs.length === 0) {
        results.innerHTML = `<p class="text-sm text-emerald-300">Add a query and at least one document line.</p>`;
        return;
      }

      searchBtn.disabled = true;
      results.innerHTML = `<p class="text-sm text-slate-400">Searching...</p>`;

      try {
        const [hits, queryMs] = await Timer.wrap(() =>
          rankBySemanticSimilarity(query, docs, useGpu, 5),
        )();

        searchMs.innerText = `${queryMs} ms`;

        if (hits.length === 0) {
          results.innerHTML = `<p class="text-sm text-slate-400">No result.</p>`;
          return;
        }

        results.innerHTML = hits
          .map(
            (hit, idx) => `
            <article class="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
              <div class="mb-1 flex items-center justify-between">
                <span class="text-xs font-bold uppercase tracking-wider text-emerald-300">#${idx + 1}</span>
                <span class="text-xs font-mono text-slate-400">score ${hit.score.toFixed(4)}</span>
              </div>
              <p class="text-sm text-slate-100">${escapeHtml(hit.text)}</p>
            </article>
          `,
          )
          .join("");
      } catch (error) {
        results.innerHTML = `<p class="text-sm text-emerald-300">Search failed: ${escapeHtml(toErrorMessage(error))}</p>`;
      } finally {
        searchBtn.disabled = false;
      }
    };

    searchBtn.addEventListener("click", runSearch);
    queryInput.addEventListener("keydown", async (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        await runSearch();
      }
    });
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
