import type { TextGenerationPipeline } from "@huggingface/transformers";
import { marked } from "marked";
import { hasGPU } from "../lib/has-gpu";
import { startLiveMs } from "../lib/live-ms";
import { renderMs } from "../lib/render-ms";
import { Timer } from "../lib/timer";
import {
    AVAILABLE_MODELS,
    Chat,
    getGenerator,
    type AvailableModel,
} from "../services/llm-service";

interface LlmDom {
    statusText: HTMLSpanElement;
    statusDot: HTMLSpanElement;
    statusPing: HTMLSpanElement;
    hardwareBadge: HTMLSpanElement;
    downloadTimer: HTMLSpanElement;
    inferenceTimer: HTMLSpanElement;
    messageList: HTMLDivElement;
    input: HTMLTextAreaElement;
    sendBtn: HTMLButtonElement;
    clearBtn: HTMLButtonElement;
    tokenInput: HTMLInputElement;
    availableModelsSelect: HTMLSelectElement;
    modelModal: HTMLDivElement;
    initialModelSelect: HTMLSelectElement;
    startModelBtn: HTMLButtonElement;
}

interface LlmState {
    generator: TextGenerationPipeline | null;
    chat: Chat | null;
    running: boolean;
    loadingModel: boolean;
    currentModel: AvailableModel | null;
    useGpu: boolean;
}

const state: LlmState = {
    generator: null,
    chat: null,
    running: false,
    loadingModel: false,
    currentModel: null,
    useGpu: false,
};

export async function render(app: HTMLElement) {
    document.title = "Local LLM Chat";

    app.innerHTML = `
		<div class="min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-300 selection:text-slate-950">
			<div class="mx-auto max-w-5xl px-4 py-6 md:px-8 md:py-10">
				<header class="mb-5 rounded-2xl border border-slate-800 bg-[radial-gradient(circle_at_top_right,_rgba(251,191,36,0.15),_transparent_45%),radial-gradient(circle_at_bottom_left,_rgba(56,189,248,0.12),_transparent_45%),rgba(15,23,42,0.85)] p-5 md:p-6">
					<div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
						<div>
							<div class="flex items-center gap-2">
								<span class="rounded-full border border-emerald-300/50 bg-emerald-300/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-200">Edge LLM</span>
                                <a href="${import.meta.env.BASE_URL}" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">back to catalog</a>
							</div>
                            <h1 class="mt-2 text-2xl font-black tracking-tight text-white md:text-4xl">Local Multi-Model Chat</h1>
							<p class="mt-2 max-w-2xl text-sm text-slate-300">Ask questions and track load and inference time with no prompt sent to a server.</p>
						</div>

						<div class="w-full max-w-xs rounded-xl border border-slate-700/70 bg-slate-900/70 p-3 text-xs font-mono">
							<div class="mb-2 flex items-center justify-between">
								<span class="text-slate-400">Runtime</span>
                                <span id="hardware-badge" class="text-emerald-300">Detecting...</span>
							</div>
							<div class="space-y-1.5">
								<div class="flex items-center justify-between">
                                    <span class="text-slate-500">Model load</span>
									<span id="download-timer" class="font-bold text-emerald-300">-- ms</span>
								</div>
								<div class="flex items-center justify-between">
                                    <span class="text-slate-500">Inference</span>
									<span id="inference-timer" class="font-bold text-emerald-300">-- ms</span>
								</div>
							</div>
						</div>
					</div>
				</header>

				<section class="mb-5 rounded-xl border border-slate-800 bg-slate-900/80 px-4 py-3">
					<div class="flex items-center gap-3">
						<span class="relative flex h-3 w-3">
							<span id="status-ping" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
							<span id="status-dot" class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
						</span>
                        <span id="status-text" class="text-xs font-bold tracking-wider text-emerald-300 uppercase">Loading model weights into local memory...</span>
					</div>
				</section>

				<main class="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_270px]">
					<section class="rounded-2xl border border-slate-800 bg-slate-900/70 p-3 md:p-4">
						<div id="message-list" class="h-[52vh] overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/85 p-3 md:p-4 space-y-3"></div>

						<div class="mt-3 space-y-2">
							<textarea
								id="llm-input"
								rows="3"
								disabled
                                placeholder="Waiting for the model to be ready..."
								class="w-full resize-none rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-emerald-300 disabled:opacity-50 disabled:cursor-not-allowed"
							></textarea>

							<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
								<div class="flex items-center gap-2 text-xs text-slate-400">
									<label for="token-input" class="font-mono">max_new_tokens</label>
                                    <input id="token-input" type="number" min="32" max="512" value="512" class="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200" />
								</div>
								<div class="flex items-center gap-2">
                                    <button id="clear-btn" class="rounded-lg border border-slate-700 px-3 py-2 text-xs font-bold uppercase tracking-wider text-slate-300 hover:border-slate-500">Clear chat</button>
                                    <button id="send-btn" disabled class="rounded-lg bg-emerald-300 px-4 py-2 text-xs font-black uppercase tracking-wider text-slate-950 disabled:opacity-45 disabled:cursor-not-allowed">Send</button>
								</div>
                                <select id="available-models" class="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200">
                                    ${AVAILABLE_MODELS.map(
                                        (model) =>
                                            `<option value="${model.key}">${model.name}</option>`,
                                    ).join("")}
                                </select>
							</div>
						</div>
					</section>

                    <aside class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                        <h2 class="text-sm font-black uppercase tracking-wider text-white">Quick prompts (EdgeAI + dev)</h2>
						<div class="mt-3 space-y-2">
                            <button data-prompt="What is your purpose?" class="quick-prompt w-full rounded-lg border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 hover:border-emerald-400/70">What is your purpose?</button>
                            <button data-prompt="What are the main limitations of running inference of Large Language Models (LLMs) on Web Browsers (Edge AI)? Be concise." class="quick-prompt w-full rounded-lg border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 hover:border-emerald-400/70">What are the limitations with using LLMs on web browsers?</button>
                            <button data-prompt="Convert this work into a checklist. Input: add login API, validate token, add tests, update docs. Output format: markdown checklist with 4 items." class="quick-prompt w-full rounded-lg border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 hover:border-emerald-400/70">Task to checklist</button>
						</div>
                        <p class="mt-4 text-[11px] text-slate-500">Responses may be slower in WASM mode.</p>
					</aside>
				</main>
			</div>

            <div id="model-modal" class="fixed inset-0 z-20 flex items-center justify-center bg-slate-950/80 px-4">
                <div class="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-900 p-5 shadow-2xl">
                    <h2 class="text-lg font-black text-white">Choose the initial model</h2>
                    <p class="mt-1 text-sm text-slate-400">Pick a model before download to avoid loading twice.</p>
                    <label for="initial-model" class="mt-4 block text-xs font-bold uppercase tracking-wider text-slate-300">Model</label>
                    <select id="initial-model" class="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100">
                        ${AVAILABLE_MODELS.map(
                            (model) =>
                                `<option value="${model.key}">${model.name}</option>`,
                        ).join("")}
                    </select>
                    <button id="start-model-btn" class="mt-4 w-full rounded-lg bg-emerald-300 px-4 py-2 text-xs font-black uppercase tracking-wider text-slate-950">Start download</button>
                </div>
            </div>
		</div>
	`;

    const dom = getDom();
    mountWelcome(dom.messageList);
    await boot(dom);
}

function getDom(): LlmDom {
    return {
        statusText: document.getElementById("status-text") as HTMLSpanElement,
        statusDot: document.getElementById("status-dot") as HTMLSpanElement,
        statusPing: document.getElementById("status-ping") as HTMLSpanElement,
        hardwareBadge: document.getElementById(
            "hardware-badge",
        ) as HTMLSpanElement,
        downloadTimer: document.getElementById(
            "download-timer",
        ) as HTMLSpanElement,
        inferenceTimer: document.getElementById(
            "inference-timer",
        ) as HTMLSpanElement,
        messageList: document.getElementById("message-list") as HTMLDivElement,
        input: document.getElementById("llm-input") as HTMLTextAreaElement,
        sendBtn: document.getElementById("send-btn") as HTMLButtonElement,
        clearBtn: document.getElementById("clear-btn") as HTMLButtonElement,
        tokenInput: document.getElementById("token-input") as HTMLInputElement,
        availableModelsSelect: document.getElementById(
            "available-models",
        ) as HTMLSelectElement,
        modelModal: document.getElementById("model-modal") as HTMLDivElement,
        initialModelSelect: document.getElementById(
            "initial-model",
        ) as HTMLSelectElement,
        startModelBtn: document.getElementById(
            "start-model-btn",
        ) as HTMLButtonElement,
    };
}
function getAvailableModelFromSelect(
    select: HTMLSelectElement,
): AvailableModel {
    const selectedKey = select.value;
    const availableModel = AVAILABLE_MODELS.find(
        (model) => model.key === selectedKey,
    );
    if (!availableModel) {
        throw new Error(`Selected model not found: ${selectedKey}`);
    }
    return availableModel;
}

async function boot(dom: LlmDom) {
    state.useGpu = await hasGPU();
    dom.hardwareBadge.innerText = state.useGpu ? "WebGPU" : "WASM";

    const selectedModel = await waitForInitialModelChoice(dom);
    dom.availableModelsSelect.value = selectedModel.key;
    await loadModel(dom, selectedModel);

    dom.sendBtn.addEventListener("click", async () => {
        await sendCurrentMessage(dom);
    });

    dom.input.addEventListener("keydown", async (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            await sendCurrentMessage(dom);
        }
    });

    dom.clearBtn.addEventListener("click", () => {
        if (!state.generator) return;

        state.chat = new Chat(state.generator);
        dom.messageList.innerHTML = "";
        mountWelcome(dom.messageList);
    });

    dom.availableModelsSelect.addEventListener("change", async () => {
        if (state.loadingModel || state.running) {
            dom.availableModelsSelect.value = state.currentModel?.key || "";
            return;
        }

        const targetModel = getAvailableModelFromSelect(
            dom.availableModelsSelect,
        );
        if (targetModel.key === state.currentModel?.key) return;

        const shouldSwitch = window.confirm(
            `Switch to ${targetModel.name}? This will reset the current chat.`,
        );
        if (!shouldSwitch) {
            dom.availableModelsSelect.value = state.currentModel?.key || "";
            return;
        }

        await loadModel(dom, targetModel);
    });

    const quickPrompts = document.querySelectorAll(
        ".quick-prompt",
    ) as NodeListOf<HTMLButtonElement>;
    quickPrompts.forEach((button) => {
        button.addEventListener("click", async () => {
            if (state.running) return;
            dom.input.value = button.dataset.prompt || "";
            await sendCurrentMessage(dom);
        });
    });
}

function waitForInitialModelChoice(dom: LlmDom): Promise<AvailableModel> {
    dom.modelModal.classList.remove("hidden");

    return new Promise((resolve) => {
        const onStart = () => {
            const model = getAvailableModelFromSelect(dom.initialModelSelect);
            dom.modelModal.classList.add("hidden");
            dom.startModelBtn.removeEventListener("click", onStart);
            resolve(model);
        };

        dom.startModelBtn.addEventListener("click", onStart);
    });
}

async function loadModel(dom: LlmDom, model: AvailableModel) {
    state.loadingModel = true;
    lockComposer(dom);
    setLoadingStatus(dom, `Loading ${model.name}...`);

    const ticker = startLiveMs(dom.downloadTimer, 20);

    try {
        const [generator, loadMs] = await Timer.wrap(() =>
            getGenerator(model),
        )();
        ticker.stop(loadMs);

        state.generator = generator;
        state.chat = new Chat(generator);
        state.currentModel = model;

        dom.availableModelsSelect.value = model.key;
        dom.messageList.innerHTML = "";
        mountWelcome(dom.messageList);

        unlockComposer(dom);
        setReadyStatus(dom, state.useGpu, model.name);
    } catch (error) {
        ticker.stop();
        setErrorStatus(dom, toErrorMessage(error));
        dom.availableModelsSelect.disabled = false;
        appendBubble(
            dom.messageList,
            "assistant",
            `Could not start the model:\n${toErrorMessage(error)}`,
        );
    } finally {
        state.loadingModel = false;
    }
}

async function sendCurrentMessage(dom: LlmDom) {
    console.debug("Sending message (page):", dom.input.value);
    const text = dom.input.value.trim();
    if (!text || !state.chat || state.running || state.loadingModel) return;

    state.running = true;
    dom.sendBtn.disabled = true;
    dom.input.disabled = true;

    const requestedTokens = Number(dom.tokenInput.value);
    const maxTokens = Number.isFinite(requestedTokens)
        ? Math.max(32, Math.min(512, Math.round(requestedTokens)))
        : 192;

    appendBubble(dom.messageList, "user", text);
    dom.input.value = "";

    const typingId = appendTypingBubble(dom.messageList);
    await Timer.wait(50);
    try {
        const [answer, responseMs] = await Timer.wrap(() =>
            state.chat!.sendMessage(text, maxTokens),
        )();

        removeTypingBubble(typingId);
        appendBubble(dom.messageList, "assistant", answer, true);
        renderMs(dom.inferenceTimer, responseMs);
    } catch (error) {
        removeTypingBubble(typingId);
        appendBubble(
            dom.messageList,
            "assistant",
            `Inference error:\n${toErrorMessage(error)}`,
        );
    } finally {
        state.running = false;
        dom.sendBtn.disabled = false;
        dom.input.disabled = false;
        dom.input.focus();
    }
}

function setReadyStatus(dom: LlmDom, usingGpu: boolean, modelName: string) {
    dom.statusText.innerText = usingGpu
        ? `${modelName} ready on WebGPU`
        : `${modelName} ready on WASM`;
    dom.statusText.className =
        "text-xs font-bold tracking-wider text-emerald-300 uppercase";
    dom.statusDot.className =
        "relative inline-flex rounded-full h-3 w-3 bg-emerald-500";
    dom.statusPing.classList.add("hidden");
}

function setLoadingStatus(dom: LlmDom, message: string) {
    dom.statusText.innerText = message;
    dom.statusText.className =
        "text-xs font-bold tracking-wider text-emerald-300 uppercase";
    dom.statusDot.className =
        "relative inline-flex rounded-full h-3 w-3 bg-emerald-500";
    dom.statusPing.classList.remove("hidden");
}

function setErrorStatus(dom: LlmDom, message: string) {
    dom.statusText.innerText = `Model load failed: ${message}`;
    dom.statusText.className =
        "text-xs font-bold tracking-wider text-emerald-300 uppercase";
    dom.statusDot.className =
        "relative inline-flex rounded-full h-3 w-3 bg-emerald-500";
    dom.statusPing.classList.add("hidden");
}

function unlockComposer(dom: LlmDom) {
    dom.input.disabled = false;
    dom.sendBtn.disabled = false;
    dom.clearBtn.disabled = false;
    dom.availableModelsSelect.disabled = false;
    dom.input.placeholder =
        "Ask about edge AI, performance, architecture, and more.";
    dom.input.focus();
}

function lockComposer(dom: LlmDom) {
    dom.input.disabled = true;
    dom.sendBtn.disabled = true;
    dom.clearBtn.disabled = true;
    dom.availableModelsSelect.disabled = true;
    dom.input.placeholder = "Waiting for the model to be ready...";
}

function mountWelcome(container: HTMLDivElement) {
    appendBubble(
        container,
        "assistant",
        "Local LLM is ready. Send a prompt or use a quick prompt.",
    );
}

function appendBubble(
    container: HTMLDivElement,
    role: "user" | "assistant",
    text: string,
    useMarkdown: boolean = false,
) {
    const row = document.createElement("div");
    row.className = `flex ${role === "user" ? "justify-end" : "justify-start"}`;

    const bubble = document.createElement("div");
    bubble.className =
        role === "user"
            ? "max-w-[88%] rounded-2xl rounded-br-md bg-emerald-300 px-3 py-2 text-sm text-slate-950"
            : "max-w-[88%] rounded-2xl rounded-bl-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100";
    if (role === "assistant" && useMarkdown) {
        bubble.className = `${bubble.className} chat-markdown`;
        bubble.innerHTML = renderAssistantMarkdown(text);
    } else {
        bubble.innerText = text;
    }

    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
}

function appendTypingBubble(container: HTMLDivElement): string {
    const id = `typing-${Date.now()}-${Math.round(Math.random() * 10_000)}`;

    const row = document.createElement("div");
    row.id = id;
    row.className = "flex justify-start";

    const bubble = document.createElement("div");
    bubble.className =
        "max-w-[88%] rounded-2xl rounded-bl-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-400";
    bubble.innerText = "Generating response...";

    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;

    return id;
}

function removeTypingBubble(id: string) {
    const el = document.getElementById(id);
    if (el) {
        el.remove();
    }
}

function toErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}

function renderAssistantMarkdown(text: string): string {
    const normalizedText = normalizeMarkdownTables(text);

    const raw = marked.parse(normalizedText, {
        async: false,
        breaks: false,
        gfm: true,
    }) as string;

    return enhanceMarkdownHtml(sanitizeHtml(raw));
}

function normalizeMarkdownTables(text: string): string {
    const lines = text.split(/\r?\n/);
    const out: string[] = [];
    let inFence = false;

    for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        if (/^\s*```/.test(line)) {
            inFence = !inFence;
            out.push(line);
            continue;
        }

        if (inFence || !line.includes("|")) {
            out.push(line);
            continue;
        }

        let j = i;
        const block: string[] = [];
        while (
            j < lines.length &&
            lines[j].includes("|") &&
            !/^\s*```/.test(lines[j])
        ) {
            block.push(lines[j]);
            j += 1;
        }

        const normalizedBlock = normalizeTableBlock(block);
        if (!normalizedBlock) {
            out.push(...block);
            i = j - 1;
            continue;
        }

        out.push(...normalizedBlock);
        i = j - 1;
    }

    return out.join("\n");
}

function normalizeTableBlock(block: string[]): string[] | null {
    if (block.length < 2) return null;

    const splitRow = (row: string) =>
        row
            .trim()
            .replace(/^\|/, "")
            .replace(/\|$/, "")
            .split("|")
            .map((cell) => cell.trim());

    const isSeparatorRow = (row: string) => {
        const cells = splitRow(row);
        return (
            cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
        );
    };

    const headerCells = splitRow(block[0]);
    if (headerCells.length < 2) return null;

    const secondIsSeparator = isSeparatorRow(block[1]);
    const dataRowsRaw = block.slice(secondIsSeparator ? 2 : 1);
    const dataRows = dataRowsRaw.map(splitRow);

    const colCount = Math.max(
        headerCells.length,
        ...dataRows.map((row) => row.length),
    );
    if (colCount < 2) return null;

    const pad = (cells: string[]) => {
        const copy = [...cells];
        while (copy.length < colCount) copy.push("");
        return copy.slice(0, colCount);
    };

    const format = (cells: string[]) => `| ${pad(cells).join(" | ")} |`;

    const output = [
        format(headerCells),
        `| ${Array.from({ length: colCount }, () => "---").join(" | ")} |`,
        ...dataRows.map(format),
    ];

    return output;
}

function enhanceMarkdownHtml(html: string): string {
    const template = document.createElement("template");
    template.innerHTML = html;

    template.content.querySelectorAll("table").forEach((table) => {
        const wrapper = document.createElement("div");
        wrapper.className = "chat-table-wrap";
        table.parentNode?.insertBefore(wrapper, table);
        wrapper.appendChild(table);
    });

    return template.innerHTML;
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

            const isScriptableSource =
                (name === "href" || name === "src" || name === "xlink:href") &&
                value.startsWith("javascript:");
            if (isScriptableSource) {
                el.removeAttribute(attr.name);
            }
        });
    });

    return template.innerHTML;
}
