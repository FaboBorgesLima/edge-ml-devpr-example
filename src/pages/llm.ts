import type { TextGenerationPipeline } from "@huggingface/transformers";
import { hasGPU } from "../lib/has-gpu";
import { startLiveMs } from "../lib/live-ms";
import { renderMs } from "../lib/render-ms";
import { Timer } from "../lib/timer";
import { Chat, getGenerator } from "../services/llm-service";

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
}

interface LlmState {
    generator: TextGenerationPipeline | null;
    chat: Chat | null;
    running: boolean;
}

const state: LlmState = {
    generator: null,
    chat: null,
    running: false,
};

export async function render(app: HTMLElement) {
    document.title = "LLM Chat Local";

    app.innerHTML = `
		<div class="min-h-screen bg-slate-950 text-slate-100 selection:bg-amber-300 selection:text-slate-950">
			<div class="mx-auto max-w-5xl px-4 py-6 md:px-8 md:py-10">
				<header class="mb-5 rounded-2xl border border-slate-800 bg-[radial-gradient(circle_at_top_right,_rgba(251,191,36,0.15),_transparent_45%),radial-gradient(circle_at_bottom_left,_rgba(56,189,248,0.12),_transparent_45%),rgba(15,23,42,0.85)] p-5 md:p-6">
					<div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
						<div>
							<div class="flex items-center gap-2">
								<span class="rounded-full border border-amber-300/50 bg-amber-300/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.2em] text-amber-200">Edge LLM</span>
								<a href="/" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">voltar ao catalogo</a>
							</div>
							<h1 class="mt-2 text-2xl font-black tracking-tight text-white md:text-4xl">Chat local com Gemma-3-270M</h1>
							<p class="mt-2 max-w-2xl text-sm text-slate-300">Pergunte, responda e acompanhe o custo de carga e inferencia sem enviar prompts para servidor.</p>
						</div>

						<div class="w-full max-w-xs rounded-xl border border-slate-700/70 bg-slate-900/70 p-3 text-xs font-mono">
							<div class="mb-2 flex items-center justify-between">
								<span class="text-slate-400">Runtime</span>
								<span id="hardware-badge" class="text-cyan-300">Detectando...</span>
							</div>
							<div class="space-y-1.5">
								<div class="flex items-center justify-between">
									<span class="text-slate-500">Carga modelo</span>
									<span id="download-timer" class="font-bold text-amber-300">-- ms</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-slate-500">Inferencia</span>
									<span id="inference-timer" class="font-bold text-cyan-300">-- ms</span>
								</div>
							</div>
						</div>
					</div>
				</header>

				<section class="mb-5 rounded-xl border border-slate-800 bg-slate-900/80 px-4 py-3">
					<div class="flex items-center gap-3">
						<span class="relative flex h-3 w-3">
							<span id="status-ping" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
							<span id="status-dot" class="relative inline-flex rounded-full h-3 w-3 bg-amber-500"></span>
						</span>
						<span id="status-text" class="text-xs font-bold tracking-wider text-amber-300 uppercase">Carregando pesos do modelo para memoria local...</span>
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
								placeholder="Aguardando o modelo ficar pronto..."
								class="w-full resize-none rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-amber-300 disabled:opacity-50 disabled:cursor-not-allowed"
							></textarea>

							<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
								<div class="flex items-center gap-2 text-xs text-slate-400">
									<label for="token-input" class="font-mono">max_new_tokens</label>
									<input id="token-input" type="number" min="32" max="1024" value="1024" class="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200" />
								</div>
								<div class="flex items-center gap-2">
									<button id="clear-btn" class="rounded-lg border border-slate-700 px-3 py-2 text-xs font-bold uppercase tracking-wider text-slate-300 hover:border-slate-500">Limpar chat</button>
									<button id="send-btn" disabled class="rounded-lg bg-amber-300 px-4 py-2 text-xs font-black uppercase tracking-wider text-slate-950 disabled:opacity-45 disabled:cursor-not-allowed">Enviar</button>
								</div>
							</div>
						</div>
					</section>

					<aside class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
						<h2 class="text-sm font-black uppercase tracking-wider text-white">Prompts rapidos</h2>
						<div class="mt-3 space-y-2">
							<button data-prompt="Explique em 3 bullets por que edge AI reduz latencia." class="quick-prompt w-full rounded-lg border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 hover:border-cyan-400/70">Por que edge AI reduz latencia?</button>
							<button data-prompt="Escreva uma mensagem de commit de Git curta e profissional (padrão Conventional Commits) para uma alteração que corrigiu uma falha de autenticação via token no middleware do Laravel." class="quick-prompt w-full rounded-lg border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 hover:border-cyan-400/70">Gere uma mensagem de commit</button>
							<button data-prompt="Liste riscos de privacidade ao usar LLM em nuvem e compare com local." class="quick-prompt w-full rounded-lg border border-slate-700 px-3 py-2 text-left text-xs text-slate-300 hover:border-cyan-400/70">Privacidade: nuvem vs local</button>
						</div>
						<p class="mt-4 text-[11px] text-slate-500">PT: respostas podem ser lentas no WASM. EN: responses can be slower in WASM mode.</p>
					</aside>
				</main>
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
    };
}

async function boot(dom: LlmDom) {
    const useGpu = await hasGPU();
    dom.hardwareBadge.innerText = useGpu ? "WebGPU" : "WASM";

    const ticker = startLiveMs(dom.downloadTimer, 20);

    try {
        const [generator, loadMs] = await Timer.wrap(() =>
            getGenerator(useGpu),
        )();
        ticker.stop(loadMs);
        state.generator = generator;
        state.chat = new Chat(generator);

        unlockComposer(dom);
        setReadyStatus(dom, useGpu);
    } catch (error) {
        ticker.stop();
        dom.statusText.innerText = `Falha ao carregar modelo: ${toErrorMessage(error)}`;
        dom.statusText.className =
            "text-xs font-bold tracking-wider text-rose-300 uppercase";
        dom.statusDot.className =
            "relative inline-flex rounded-full h-3 w-3 bg-rose-500";
        dom.statusPing.remove();
        appendBubble(
            dom.messageList,
            "assistant",
            `Nao foi possivel iniciar o modelo:\n${toErrorMessage(error)}`,
        );
        return;
    }

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

async function sendCurrentMessage(dom: LlmDom) {
    console.debug("Sending message (page):", dom.input.value);
    const text = dom.input.value.trim();
    if (!text || !state.chat || state.running) return;

    state.running = true;
    dom.sendBtn.disabled = true;
    dom.input.disabled = true;

    const requestedTokens = Number(dom.tokenInput.value);
    const maxTokens = Number.isFinite(requestedTokens)
        ? Math.max(32, Math.min(1024, Math.round(requestedTokens)))
        : 256;

    appendBubble(dom.messageList, "user", text);
    dom.input.value = "";

    const typingId = appendTypingBubble(dom.messageList);
    await Timer.wait(50);
    try {
        const [answer, responseMs] = await Timer.wrap(() =>
            state.chat!.sendMessage(text, maxTokens),
        )();

        removeTypingBubble(typingId);
        appendBubble(dom.messageList, "assistant", answer);
        renderMs(dom.inferenceTimer, responseMs);
    } catch (error) {
        removeTypingBubble(typingId);
        appendBubble(
            dom.messageList,
            "assistant",
            `Erro de inferencia:\n${toErrorMessage(error)}`,
        );
    } finally {
        state.running = false;
        dom.sendBtn.disabled = false;
        dom.input.disabled = false;
        dom.input.focus();
    }
}

function setReadyStatus(dom: LlmDom, usingGpu: boolean) {
    dom.statusText.innerText = usingGpu
        ? "Modelo pronto em WebGPU"
        : "Modelo pronto em WASM";
    dom.statusText.className =
        "text-xs font-bold tracking-wider text-emerald-300 uppercase";
    dom.statusDot.className =
        "relative inline-flex rounded-full h-3 w-3 bg-emerald-500";
    dom.statusPing.remove();
}

function unlockComposer(dom: LlmDom) {
    dom.input.disabled = false;
    dom.sendBtn.disabled = false;
    dom.input.placeholder =
        "Pergunte algo sobre Edge AI, performance, arquitetura, etc.";
    dom.input.focus();
}

function mountWelcome(container: HTMLDivElement) {
    appendBubble(
        container,
        "assistant",
        "LLM local pronto para demo. Envie um prompt ou use um atalho na lateral.",
    );
}

function appendBubble(
    container: HTMLDivElement,
    role: "user" | "assistant",
    text: string,
) {
    const row = document.createElement("div");
    row.className = `flex ${role === "user" ? "justify-end" : "justify-start"}`;

    const bubble = document.createElement("div");
    bubble.className =
        role === "user"
            ? "max-w-[88%] rounded-2xl rounded-br-md bg-amber-300 px-3 py-2 text-sm text-slate-950"
            : "max-w-[88%] rounded-2xl rounded-bl-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100";
    bubble.innerText = text;

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
    bubble.innerText = "Gerando resposta...";

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
