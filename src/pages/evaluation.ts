import {
    getEvaluationPipeline,
    getEvaluation,
} from "../services/evaluation-service";
import { Timer } from "../lib/timer";
import { renderMs } from "../lib/render-ms";
import { startLiveMs } from "../lib/live-ms";
import { resizeTextArea } from "../lib/resize-text-area";
import { TextClassificationPipeline } from "@huggingface/transformers";
import { hasGPU } from "../lib/has-gpu";

export async function render(app: HTMLElement) {
    document.title = "Local Sentiment Analysis";

    app.innerHTML = `
        <div class="min-h-screen bg-slate-950 text-slate-100 p-6 selection:bg-emerald-500 selection:text-white">
            <div class="mx-auto w-full max-w-5xl space-y-5">
                <header class="rounded-2xl border border-slate-800 bg-[radial-gradient(circle_at_top_right,_rgba(99,102,241,0.14),_transparent_45%),radial-gradient(circle_at_bottom_left,_rgba(56,189,248,0.1),_transparent_45%),rgba(15,23,42,0.85)] p-5 md:p-6">
                    <div class="flex items-center justify-between gap-4">
                        <div>
                            <div class="flex items-center gap-2">
                                <span class="rounded-full border border-emerald-300/50 bg-emerald-300/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-200">Edge Eval</span>
                                <a href="${import.meta.env.BASE_URL}" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">back to catalog</a>
                            </div>
                            <h1 class="mt-2 text-2xl font-black tracking-tight text-white md:text-3xl">Local Sentiment Analysis</h1>
                        </div>
                    </div>
                </header>

                <div class="flex flex-col items-center">
            <div class="w-full max-w-3xl bg-slate-900/80 border border-slate-800 rounded-2xl shadow-2xl backdrop-blur-xl overflow-hidden">
                
                <div class="bg-slate-950/80 px-6 py-3.5 border-b border-slate-800/80 flex justify-between items-center">
                    <div class="flex items-center gap-3">
                        <span class="relative flex h-3 w-3">
                            <span id="status-ping" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span id="status-dot" class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                        </span>
                        <span id="status-text" class="text-xs font-bold tracking-wider text-emerald-400 uppercase">
                            Allocating tensors in memory...
                        </span>
                    </div>
                    
                    <div class="flex gap-2 text-[11px] font-mono text-slate-400">
                        <span id="hardware-badge" class="bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">WASM</span>
                        <span class="bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">Single-Thread</span>
                    </div>
                </div>

                <div class="p-8 space-y-6">
                    
                    <div>
                        <label for="evaluation-input" class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                            Text Input:
                        </label>
                        <textarea 
                            id="evaluation-input"
                            rows="2"
                            disabled
                            placeholder="Waiting for the model to load..."
                            class="w-full bg-slate-950/90 border border-slate-700/80 rounded-xl p-4 text-2xl font-medium text-slate-100 placeholder-slate-600 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all resize-none disabled:opacity-40 disabled:cursor-not-allowed"
                        ></textarea>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-12 gap-4 pt-2">
                        
                        <div class="md:col-span-7 bg-slate-950/60 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center min-h-35 relative overflow-hidden group">
                            <div class="absolute inset-0 bg-linear-to-br from-emerald-500/5 to-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            
                            <span id="sentiment-emoji" class="text-6xl mb-2 select-none transform transition-transform duration-150 group-hover:scale-110">🤖</span>
                            <div id="evaluation-result" class="text-lg font-bold tracking-tight text-slate-300 text-center">
                                Waiting for input...
                            </div>
                        </div>

                        <div class="md:col-span-5 bg-slate-950/60 border border-slate-800 rounded-xl p-5 flex flex-col justify-between font-mono">
                            <div class="text-[11px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-2 flex justify-between items-center">
                                <span>Telemetria</span>
                                <span class="text-emerald-400 text-[9px]">● LOCAL</span>
                            </div>
                            
                            <div class="space-y-3 my-auto py-2 text-xs">
                                <div class="flex justify-between items-center">
                                    <span class="text-slate-400">Download/RAM:</span>
                                    <span id="download-timer" class="font-bold text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-2 py-0.5 rounded">
                                        -- ms
                                    </span>
                                </div>
                                
                                <div class="flex justify-between items-center">
                                    <span class="text-slate-400">Response Time:</span>
                                    <span id="evaluation-timer" class="font-bold text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-2 py-0.5 rounded">
                                        -- ms
                                    </span>
                                </div>
                            </div>

                            <div class="text-[10px] text-slate-500 text-right border-t border-slate-800 pt-2">
                                0.0 bytes sent over network
                            </div>
                        </div>

                    </div>

                </div>

            </div>

            <p class="text-slate-600 text-xs mt-4 font-mono">
                Powered by @huggingface/transformers (v3) & WebAssembly
            </p>
                </div>
            </div>
        </div>
    `;

    await boot();
}

async function boot() {
    const textInput = document.getElementById(
        "evaluation-input",
    ) as HTMLTextAreaElement;
    const evaluationResult = document.getElementById(
        "evaluation-result",
    ) as HTMLDivElement;
    const downloadTimerDiv = document.getElementById(
        "download-timer",
    ) as HTMLSpanElement;
    const evaluationTimerDiv = document.getElementById(
        "evaluation-timer",
    ) as HTMLSpanElement;
    const hardwareBadge = document.getElementById(
        "hardware-badge",
    ) as HTMLSpanElement;

    hardwareBadge.innerText = (await hasGPU()) ? "WebGPU" : "WASM";

    const statusText = document.getElementById(
        "status-text",
    ) as HTMLSpanElement;
    const statusDot = document.getElementById("status-dot") as HTMLSpanElement;
    const statusPing = document.getElementById(
        "status-ping",
    ) as HTMLSpanElement;

    // Keep input locked while the model is being allocated.
    textInput.disabled = true;

    const loadingTicker = startLiveMs(downloadTimerDiv, 20);
    const [pipe, downloadTime] = await Timer.wrap(async () =>
        getEvaluationPipeline(await hasGPU()),
    )();
    loadingTicker.stop(downloadTime);

    renderMs(downloadTimerDiv, downloadTime);

    statusText.innerText = "Model fully loaded in RAM";
    statusText.className =
        "text-xs font-bold tracking-wider text-emerald-400 uppercase";
    statusDot.className =
        "relative inline-flex rounded-full h-3 w-3 bg-emerald-500";
    statusPing.remove();

    textInput.disabled = false;
    textInput.placeholder =
        "Example: The Wi-Fi at this event is surprisingly fast...";
    textInput.focus();

    textInput.addEventListener("input", async (event) => {
        const target = event.target as HTMLTextAreaElement;
        resizeTextArea(target);

        const text = target.value;

        if (!text.trim()) {
            evaluationResult.innerText = "Waiting for input...";
            document.getElementById("sentiment-emoji")!.innerText = "🤖";
            evaluationTimerDiv.innerText = "-- ms";
            return;
        }

        await updateEvaluation(
            evaluationResult,
            evaluationTimerDiv,
            text,
            pipe,
        );
    });
}

async function updateEvaluation(
    evaluationResult: HTMLDivElement,
    timerDiv: HTMLSpanElement,
    evaluation: string,
    pipeline: TextClassificationPipeline,
) {
    const [result, responseTime] = await Timer.wrap(() =>
        getEvaluation(evaluation, pipeline),
    )();

    renderMs(timerDiv, responseTime);

    const scoreFormatado = (result.score * 100).toFixed(1);
    evaluationResult.innerText = `${result.label} (${scoreFormatado}% confidence)`;

    // Simple class-to-emoji mapping for live stage feedback.
    const emojiEl = document.getElementById(
        "sentiment-emoji",
    ) as HTMLSpanElement;
    const lbl = result.label.toLowerCase();

    if (lbl.includes("5") || lbl.includes("pos") || lbl.includes("star5"))
        emojiEl.innerText = "🤩";
    else if (lbl.includes("4")) emojiEl.innerText = "🙂";
    else if (lbl.includes("3") || lbl.includes("neu")) emojiEl.innerText = "😐";
    else if (lbl.includes("2")) emojiEl.innerText = "🙁";
    else if (lbl.includes("1") || lbl.includes("neg") || lbl.includes("star1"))
        emojiEl.innerText = "🤬";
}
