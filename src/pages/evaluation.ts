import {
    getEvaluationPipeline,
    getEvaluation,
} from "../services/evaluation-service";
import { Timer } from "../lib/timer";
import { renderMs } from "../lib/render-ms";
import { resizeTextArea } from "../lib/resize-text-area";
import { TextClassificationPipeline } from "@huggingface/transformers";

export async function render(app: HTMLElement) {
    app.innerHTML = `
        <div class="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 font-sans selection:bg-indigo-500 selection:text-white">
            
            <div class="w-full max-w-3xl bg-slate-900/80 border border-slate-800 rounded-2xl shadow-2xl backdrop-blur-xl overflow-hidden">
                
                <div class="bg-slate-950/80 px-6 py-3.5 border-b border-slate-800/80 flex justify-between items-center">
                    <div class="flex items-center gap-3">
                        <span class="relative flex h-3 w-3">
                            <span id="status-ping" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                            <span id="status-dot" class="relative inline-flex rounded-full h-3 w-3 bg-amber-500"></span>
                        </span>
                        <span id="status-text" class="text-xs font-bold tracking-wider text-amber-400 uppercase">
                            Alocando Tensores na Memória...
                        </span>
                    </div>
                    
                    <div class="flex gap-2 text-[11px] font-mono text-slate-400">
                        <span class="bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">WASM</span>
                        <span class="bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">Single-Thread</span>
                    </div>
                </div>

                <div class="p-8 space-y-6">
                    
                    <div>
                        <label for="evaluation-input" class="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                            Entrada de Texto:
                        </label>
                        <textarea 
                            id="evaluation-input"
                            rows="2"
                            disabled
                            placeholder="Aguardando o modelo ser carregado..."
                            class="w-full bg-slate-950/90 border border-slate-700/80 rounded-xl p-4 text-2xl font-medium text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all resize-none disabled:opacity-40 disabled:cursor-not-allowed"
                        ></textarea>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-12 gap-4 pt-2">
                        
                        <div class="md:col-span-7 bg-slate-950/60 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center min-h-35 relative overflow-hidden group">
                            <div class="absolute inset-0 bg-linear-to-br from-indigo-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                            
                            <span id="sentiment-emoji" class="text-6xl mb-2 select-none transform transition-transform duration-150 group-hover:scale-110">🤖</span>
                            <div id="evaluation-result" class="text-lg font-bold tracking-tight text-slate-300 text-center">
                                Aguardando estímulo...
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
                                    <span id="download-timer" class="font-bold text-amber-400 bg-amber-950/50 border border-amber-800/60 px-2 py-0.5 rounded">
                                        -- ms
                                    </span>
                                </div>
                                
                                <div class="flex justify-between items-center">
                                    <span class="text-slate-400">Tempo de Resposta:</span>
                                    <span id="evaluation-timer" class="font-bold text-indigo-400 bg-indigo-950/50 border border-indigo-800/60 px-2 py-0.5 rounded">
                                        -- ms
                                    </span>
                                </div>
                            </div>

                            <div class="text-[10px] text-slate-500 text-right border-t border-slate-800 pt-2">
                                0.0 bytes trafegados na rede
                            </div>
                        </div>

                    </div>

                </div>

            </div>

            <p class="text-slate-600 text-xs mt-4 font-mono">
                Powered by @huggingface/transformers (v3) & WebAssembly
            </p>
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

    const statusText = document.getElementById(
        "status-text",
    ) as HTMLSpanElement;
    const statusDot = document.getElementById("status-dot") as HTMLSpanElement;
    const statusPing = document.getElementById(
        "status-ping",
    ) as HTMLSpanElement;

    // Trava a UI enquanto baixa
    textInput.disabled = true;

    const [pipe, downloadTime] = await Timer.wrap(getEvaluationPipeline)();

    renderMs(downloadTimerDiv, downloadTime);

    // O MOMENTO "MIC DROP" DA INICIALIZAÇÃO:
    statusText.innerText = "Modelo 100% Carregado na RAM";
    statusText.className =
        "text-xs font-bold tracking-wider text-emerald-400 uppercase";
    statusDot.className =
        "relative inline-flex rounded-full h-3 w-3 bg-emerald-500";
    statusPing.remove(); // Mata o efeito piscante

    textInput.disabled = false;
    textInput.placeholder =
        "Ex: O Wi-Fi deste evento está surpreendentemente rápido...";
    textInput.focus(); // <--- ATENÇÃO AQUI: Foco automático para você não caçar o mouse no palco

    textInput.addEventListener("input", async (event) => {
        const target = event.target as HTMLTextAreaElement;
        resizeTextArea(target);

        const text = target.value;

        if (!text.trim()) {
            evaluationResult.innerText = "Aguardando estímulo...";
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

    // Atualiza o texto
    const scoreFormatado = (result.score * 100).toFixed(1);
    evaluationResult.innerText = `${result.label} (${scoreFormatado} de certeza)`;

    // Atualiza o Emoji de Palco baseado na label do BERT
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
