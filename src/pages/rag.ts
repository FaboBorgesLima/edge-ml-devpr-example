import { marked } from "marked";
import { hasGPU } from "../lib/has-gpu";
import { Timer } from "../lib/timer";
import { AVAILABLE_MODELS, type AvailableModel } from "../services/llm-service";
import { createRagRuntime, type RagRuntime } from "../services/rag-service";

const SAMPLE_KB = [
    "WebGPU is usually faster for larger models but requires compatible GPU drivers.",
    "WASM is more compatible across browsers and can run without GPU support.",
    "RAG combines retrieval and generation: retrieve relevant snippets before answering.",
    "Chunking documents into small passages improves retrieval quality.",
    "For private data, running retrieval and generation locally reduces exposure risk.",
].join("\n");

interface RagState {
    runtime: RagRuntime | null;
    useGpu: boolean;
    loading: boolean;
    currentModel: AvailableModel | null;
}

const state: RagState = {
    runtime: null,
    useGpu: false,
    loading: false,
    currentModel: null,
};

export async function render(app: HTMLElement) {
    document.title = "Local RAG";

    app.innerHTML = `
    <div class="min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-300 selection:text-slate-950">
      <div class="mx-auto max-w-6xl px-4 py-6 md:px-8 md:py-10">
        <header class="mb-5 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 md:p-6">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="flex items-center gap-2">
                <span class="rounded-full border border-emerald-300/50 bg-emerald-300/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-200">Edge RAG</span>
                <a href="${import.meta.env.BASE_URL}" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">back to catalog</a>
              </div>
              <h1 class="mt-2 text-2xl font-black tracking-tight text-white md:text-4xl">RAG (Semantic Search + LLM)</h1>
              <p class="mt-2 max-w-3xl text-sm text-slate-300">Retrieve relevant snippets with embeddings, then answer with a local LLM using only retrieved context.</p>
            </div>
            <div class="rounded-xl border border-slate-700/70 bg-slate-950/60 p-3 text-xs font-mono">
              <div class="flex items-center justify-between gap-4"><span class="text-slate-400">Runtime</span><span id="hardware-badge" class="text-emerald-300">Detecting...</span></div>
              <div class="mt-2 flex items-center justify-between gap-4"><span class="text-slate-500">Model load</span><span id="load-ms" class="font-bold text-emerald-300">-- ms</span></div>
              <div class="mt-1 flex items-center justify-between gap-4"><span class="text-slate-500">Answer</span><span id="answer-ms" class="font-bold text-emerald-300">-- ms</span></div>
            </div>
          </div>
        </header>

        <main class="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px]">
          <section class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <label for="kb-input" class="text-xs font-bold uppercase tracking-wider text-slate-300">Knowledge base (one line per snippet)</label>
            <textarea id="kb-input" rows="7" class="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-emerald-300 focus:outline-none">${SAMPLE_KB}</textarea>

            <div class="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-[1fr_200px]">
              <input id="question-input" class="rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-emerald-300 focus:outline-none" placeholder="Ask a question about the knowledge base" />
              <select id="model-select" class="rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100">
                ${AVAILABLE_MODELS.map(
                    (model) =>
                        `<option value="${model.key}">${model.name}</option>`,
                ).join("")}
              </select>
            </div>

            <div class="mt-3 flex items-center gap-2">
              <button id="ask-btn" class="rounded-lg bg-emerald-300 px-4 py-2 text-xs font-black uppercase tracking-wider text-slate-950">Ask with RAG</button>
            </div>

            <div class="mt-4 space-y-3">
              <div>
                <h2 class="text-xs font-bold uppercase tracking-wider text-emerald-300">Retrieved Context</h2>
                <div id="retrieved" class="mt-2 space-y-2"></div>
              </div>
              <div>
                <h2 class="text-xs font-bold uppercase tracking-wider text-emerald-300">Answer</h2>
                <div id="answer" class="chat-markdown mt-2 rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-sm text-slate-100"></div>
              </div>
            </div>
          </section>

          <aside class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <h2 class="text-sm font-black uppercase tracking-wider text-white">How It Works</h2>
            <ol class="mt-3 list-decimal space-y-2 pl-4 text-xs text-slate-300">
              <li>Embed question and snippets locally.</li>
              <li>Retrieve top semantic matches.</li>
              <li>Send only retrieved snippets to the LLM.</li>
              <li>Generate grounded answer.</li>
            </ol>
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
    const answerMs = document.getElementById("answer-ms") as HTMLSpanElement;
    const modelSelect = document.getElementById("model-select") as HTMLSelectElement;
    const kbInput = document.getElementById("kb-input") as HTMLTextAreaElement;
    const questionInput = document.getElementById("question-input") as HTMLInputElement;
    const askBtn = document.getElementById("ask-btn") as HTMLButtonElement;
    const retrieved = document.getElementById("retrieved") as HTMLDivElement;
    const answer = document.getElementById("answer") as HTMLDivElement;

    const defaultModel = AVAILABLE_MODELS.find((m) => m.key.includes("135M"));
    if (defaultModel) modelSelect.value = defaultModel.key;

    state.useGpu = await hasGPU();
    hardwareBadge.innerText = state.useGpu ? "WebGPU" : "WASM";

    await ensureRuntime(modelSelect.value, loadMs);

    modelSelect.addEventListener("change", async () => {
        await ensureRuntime(modelSelect.value, loadMs);
    });

    const ask = async () => {
        const question = questionInput.value.trim();
        const docs = kbInput.value
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter((line) => line.length > 0);

        if (!question || docs.length === 0) {
            answer.innerHTML = `<p class="text-emerald-300">Add a question and at least one snippet.</p>`;
            return;
        }

        if (!state.runtime) {
            answer.innerHTML = `<p class="text-emerald-300">Runtime is not ready.</p>`;
            return;
        }

        askBtn.disabled = true;
        answer.innerHTML = `<p class="text-slate-400">Generating answer...</p>`;
        retrieved.innerHTML = "";

        try {
            const [result, ms] = await Timer.wrap(() =>
                state.runtime!.ask(question, docs, 220),
            )();
            answerMs.innerText = `${ms} ms`;

            retrieved.innerHTML = result.retrieved
                .map(
                    (hit, idx) => `
                    <article class="rounded-lg border border-slate-800 bg-slate-950/70 p-2">
                      <div class="mb-1 flex items-center justify-between">
                        <span class="text-[10px] font-bold uppercase tracking-wider text-emerald-300">#${idx + 1}</span>
                        <span class="text-[10px] font-mono text-slate-400">${hit.score.toFixed(4)}</span>
                      </div>
                      <p class="text-xs text-slate-200">${escapeHtml(hit.text)}</p>
                    </article>
                `,
                )
                .join("");

            answer.innerHTML = sanitizeHtml(
                marked.parse(result.answer, {
                    async: false,
                    breaks: false,
                    gfm: true,
                }) as string,
            );
        } catch (error) {
            answer.innerHTML = `<p class="text-emerald-300">RAG failed: ${escapeHtml(toErrorMessage(error))}</p>`;
        } finally {
            askBtn.disabled = false;
        }
    };

    askBtn.addEventListener("click", ask);
    questionInput.addEventListener("keydown", async (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            await ask();
        }
    });
}

async function ensureRuntime(modelKey: string, loadMs: HTMLSpanElement) {
    const model = AVAILABLE_MODELS.find((m) => m.key === modelKey);
    if (!model) return;

    if (state.loading) return;
    if (state.currentModel?.key === model.key && state.runtime) return;

    state.loading = true;
    const [runtime, ms] = await Timer.wrap(() =>
        createRagRuntime(state.useGpu, model),
    )();
    state.runtime = runtime;
    state.currentModel = model;
    loadMs.innerText = `${ms} ms`;
    state.loading = false;
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

function sanitizeHtml(html: string): string {
    const template = document.createElement("template");
    template.innerHTML = html;

    const blockedTags = new Set([
        "script",
        "style",
        "iframe",
        "object",
        "embed",
        "link",
        "meta",
    ]);

    template.content.querySelectorAll("*").forEach((el) => {
        const tagName = el.tagName.toLowerCase();
        if (blockedTags.has(tagName)) {
            el.remove();
            return;
        }

        Array.from(el.attributes).forEach((attr) => {
            const name = attr.name.toLowerCase();
            const value = attr.value.trim().toLowerCase();
            if (name.startsWith("on")) {
                el.removeAttribute(attr.name);
                return;
            }
            const scriptUrl =
                (name === "href" || name === "src" || name === "xlink:href") &&
                value.startsWith("javascript:");
            if (scriptUrl) {
                el.removeAttribute(attr.name);
            }
        });
    });

    return template.innerHTML;
}
