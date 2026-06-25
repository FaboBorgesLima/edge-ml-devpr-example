import type { PreTrainedModel, Processor } from "@huggingface/transformers";
import { startLiveMs } from "../lib/live-ms";
import {
    getVlm,
    runFreeInspection,
    runMasterScan,
    type GroundedFinding,
    type MasterScanResult,
    type ScanPanel,
    type TaskScanResult,
} from "../services/vlm-service";
import { hasGPU } from "../lib/has-gpu";

interface VlmState {
    processor: Processor | null;
    model: PreTrainedModel | null;
    currentFile: File | null;
    imageBitmap: ImageBitmap | null;
    masterScan: MasterScanResult | null;
    freeInspection: TaskScanResult | null;
    activePanel: ScanPanel;
    highlightedFindingId: string | null;
}

interface VlmDom {
    fileInput: HTMLInputElement;
    dropArea: HTMLLabelElement;
    uploadPrompt: HTMLDivElement;
    imageCanvas: HTMLCanvasElement;
    overlayLayer: HTMLDivElement;
    scanStage: HTMLSpanElement;
    scanProgressBar: HTMLDivElement;
    hardwareBadge: HTMLSpanElement;
    downloadTimer: HTMLSpanElement;
    evaluationTimer: HTMLSpanElement;
    reportContainer: HTMLDivElement;
    panelButtons: Record<
        Exclude<ScanPanel, "free-inspection">,
        HTMLButtonElement
    >;
    freeQueryInput: HTMLInputElement;
    freeQueryForm: HTMLFormElement;
    quickAlertButtons: NodeListOf<HTMLButtonElement>;
}

const state: VlmState = {
    processor: null,
    model: null,
    currentFile: null,
    imageBitmap: null,
    masterScan: null,
    freeInspection: null,
    activePanel: "description",
    highlightedFindingId: null,
};

export async function render(app: HTMLElement) {
    document.title =
        "Microsoft Florence-2 - Master Scan and Open Inspection in Browser";

    app.innerHTML = `
        <div class="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 selection:bg-emerald-400 selection:text-slate-950">
            <div class="max-w-6xl mx-auto space-y-6">
                <header class="border-b border-slate-800 pb-5 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.2em] bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 rounded">Edge VLM</span>
                            <a href="${import.meta.env.BASE_URL}" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">back to catalog</a>
                        </div>
                        <div class="mt-2 flex items-center gap-2">
                            <h1 class="text-2xl md:text-3xl font-black text-white tracking-tight">Florence-2 Master Scan</h1>
                        </div>
                        <p class="text-slate-400 text-sm mt-1">One heavy pass per image, instant panels, and bidirectional grounding.</p>
                    </div>
                    <div class="grid grid-cols-2 gap-3 text-xs font-mono bg-slate-900/70 p-3 rounded-xl border border-slate-800 min-w-[280px]">
                        <div>
                            <div class="text-slate-500 uppercase text-[10px] tracking-wider">Model Load</div>
                            <div id="download-timer" class="text-emerald-400 font-bold text-base">Time: --ms</div>
                        </div>
                        <div>
                            <div class="text-slate-500 uppercase text-[10px] tracking-wider">Master Scan</div>
                            <div id="evaluation-timer" class="text-emerald-300 font-bold text-base">Time: --ms</div>
                        </div>
                    </div>
                </header>

                <section class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 md:p-5 space-y-3">
                    <div class="flex items-center justify-between gap-3 flex-wrap">
                        <span id="scan-stage" class="text-xs md:text-sm font-bold tracking-wide text-emerald-300 uppercase">Initializing model...</span>
                        <span class="text-[10px] font-mono px-2 py-1 rounded border border-emerald-900/50 bg-emerald-950/40 text-emerald-300">
                            <span id="hardware-badge">Detecting hardware...</span>
                        </span>
                    </div>
                    <div class="h-2 rounded-full bg-slate-800 overflow-hidden">
                        <div id="scan-progress-bar" class="h-full w-0 bg-linear-to-r from-emerald-400 to-emerald-400 transition-all duration-200"></div>
                    </div>
                    <div class="text-[11px] text-slate-500">The expensive tensor pass runs once per image.</div>
                </section>

                <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
                    <div class="space-y-4">
                        <label id="drop-area" class="block relative border-2 border-dashed border-slate-700 hover:border-emerald-400/70 rounded-2xl bg-slate-900/60 p-4 cursor-pointer transition-colors">
                            <input id="file-input" type="file" accept="image/*" class="hidden" />
                            <div id="upload-prompt" class="text-center py-12 space-y-2">
                                <div class="text-5xl">🛰️</div>
                                <div class="font-bold text-emerald-300 text-sm">Drop an image to start Master Scan</div>
                                <p class="text-slate-500 text-xs">After the base scan, all panels respond instantly.</p>
                            </div>
                            <canvas id="image-canvas" class="hidden w-full h-auto rounded-xl border border-slate-700/60 bg-slate-950"></canvas>
                            <div id="overlay-layer" class="hidden absolute left-4 right-4 pointer-events-none"></div>
                        </label>

                        <form id="free-query-form" class="space-y-2">
                            <label class="block text-[11px] font-bold uppercase tracking-wider text-slate-400">Open Inspection / Open Vocabulary</label>
                            <div class="flex gap-2">
                                <input
                                    id="free-query-input"
                                    type="text"
                                    class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-emerald-400"
                                    placeholder="What should be inspected? Example: rusty screws"
                                    disabled
                                />
                                <button type="submit" class="px-3 py-2 rounded-lg bg-emerald-500 text-slate-950 text-xs font-bold uppercase tracking-wide disabled:opacity-40" disabled id="free-query-submit">
                                    Inspect
                                </button>
                            </div>
                            <div class="flex flex-wrap gap-2">
                                <button type="button" data-alert="people without safety helmets" class="quick-alert px-2.5 py-1 rounded-full text-[11px] border border-slate-700 text-slate-300 hover:border-emerald-400/60">Helmet check</button>
                                <button type="button" data-alert="readable barcode" class="quick-alert px-2.5 py-1 rounded-full text-[11px] border border-slate-700 text-slate-300 hover:border-emerald-400/60">Barcode check</button>
                                <button type="button" data-alert="open windows" class="quick-alert px-2.5 py-1 rounded-full text-[11px] border border-slate-700 text-slate-300 hover:border-emerald-400/60">Open windows</button>
                            </div>
                        </form>
                    </div>

                    <div class="rounded-2xl border border-slate-800 bg-slate-900/70 overflow-hidden flex flex-col min-h-[420px]">
                        <div class="p-3 border-b border-slate-800 bg-slate-950/60">
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                                <button id="panel-description" class="panel-btn px-2 py-2 rounded-lg text-[11px] font-bold bg-emerald-500 text-slate-950">General Description</button>
                                <button id="panel-ocr" class="panel-btn px-2 py-2 rounded-lg text-[11px] font-bold text-slate-300 border border-slate-700">Read Text (OCR)</button>
                                <button id="panel-detection" class="panel-btn px-2 py-2 rounded-lg text-[11px] font-bold text-slate-300 border border-slate-700">Detailed Detection</button>
                                <button id="panel-grid" class="panel-btn px-2 py-2 rounded-lg text-[11px] font-bold text-slate-300 border border-slate-700">Grid Scan</button>
                            </div>
                        </div>
                        <div id="report-container" class="p-4 md:p-5 text-sm text-slate-200 whitespace-pre-wrap overflow-y-auto flex-1">
                            Waiting for model...
                        </div>
                    </div>
                </section>
            </div>
        </div>
    `;

    await boot();
}

async function boot() {
    const dom = getDomRefs();

    state.activePanel = "description";
    state.highlightedFindingId = null;

    // Show the backend in use to explain latency differences to the audience.
    dom.hardwareBadge.innerText = (await hasGPU())
        ? "WebGPU active"
        : "WASM active";

    dom.fileInput.disabled = true;
    setStage(dom, "Loading Florence-2 into local memory...", 0.06);

    const downloadTicker = startLiveMs(dom.downloadTimer, 20);

    try {
        const [processor, model] = await getVlm(await hasGPU());
        state.processor = processor;
        state.model = model;
        downloadTicker.stop();

        setStage(dom, "Model ready. Upload an image to run Master Scan.", 0.12);

        dom.fileInput.disabled = false;
        updateReport(
            dom,
            "Waiting for an image. Panels will be filled after the base scan.",
        );

        wireUpload(dom);
        wirePanels(dom);
        wireFreeInspection(dom);
    } catch (error) {
        downloadTicker.stop();
        updateReport(
            dom,
            `Failed to initialize Florence-2:\n${toErrorMessage(error)}`,
        );
        setStage(dom, "Initialization failed", 1);
    }
}

function getDomRefs(): VlmDom {
    return {
        fileInput: document.getElementById("file-input") as HTMLInputElement,
        dropArea: document.getElementById("drop-area") as HTMLLabelElement,
        uploadPrompt: document.getElementById(
            "upload-prompt",
        ) as HTMLDivElement,
        imageCanvas: document.getElementById(
            "image-canvas",
        ) as HTMLCanvasElement,
        overlayLayer: document.getElementById(
            "overlay-layer",
        ) as HTMLDivElement,
        scanStage: document.getElementById("scan-stage") as HTMLSpanElement,
        scanProgressBar: document.getElementById(
            "scan-progress-bar",
        ) as HTMLDivElement,
        hardwareBadge: document.getElementById(
            "hardware-badge",
        ) as HTMLSpanElement,
        downloadTimer: document.getElementById(
            "download-timer",
        ) as HTMLSpanElement,
        evaluationTimer: document.getElementById(
            "evaluation-timer",
        ) as HTMLSpanElement,
        reportContainer: document.getElementById(
            "report-container",
        ) as HTMLDivElement,
        panelButtons: {
            description: document.getElementById(
                "panel-description",
            ) as HTMLButtonElement,
            ocr: document.getElementById("panel-ocr") as HTMLButtonElement,
            detection: document.getElementById(
                "panel-detection",
            ) as HTMLButtonElement,
            grid: document.getElementById("panel-grid") as HTMLButtonElement,
        },
        freeQueryInput: document.getElementById(
            "free-query-input",
        ) as HTMLInputElement,
        freeQueryForm: document.getElementById(
            "free-query-form",
        ) as HTMLFormElement,
        quickAlertButtons: document.querySelectorAll(
            ".quick-alert",
        ) as NodeListOf<HTMLButtonElement>,
    };
}

function wireUpload(dom: VlmDom) {
    dom.fileInput.addEventListener("change", async (event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (!file) return;
        await executeMasterScan(dom, file);
    });

    dom.dropArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        dom.dropArea.classList.add("border-emerald-400");
    });

    dom.dropArea.addEventListener("dragleave", () => {
        dom.dropArea.classList.remove("border-emerald-400");
    });

    dom.dropArea.addEventListener("drop", async (event) => {
        event.preventDefault();
        dom.dropArea.classList.remove("border-emerald-400");
        const file = event.dataTransfer?.files?.[0];
        if (!file || !file.type.startsWith("image/")) return;

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        dom.fileInput.files = dataTransfer.files;

        await executeMasterScan(dom, file);
    });

    dom.imageCanvas.addEventListener("mousemove", (event) => {
        const visibleFindings = getVisibleFindings();
        const hoveredId = findHoveredFindingId(
            dom.imageCanvas,
            event,
            visibleFindings,
        );
        if (hoveredId !== state.highlightedFindingId) {
            state.highlightedFindingId = hoveredId;
            syncHighlightState(dom);
        }
    });

    dom.imageCanvas.addEventListener("mouseleave", () => {
        if (state.highlightedFindingId !== null) {
            state.highlightedFindingId = null;
            syncHighlightState(dom);
        }
    });
}

function wirePanels(dom: VlmDom) {
    const panels: Array<Exclude<ScanPanel, "free-inspection">> = [
        "description",
        "ocr",
        "detection",
        "grid",
    ];

    panels.forEach((panel) => {
        dom.panelButtons[panel].addEventListener("click", () => {
            state.activePanel = panel;
            state.highlightedFindingId = null;
            state.freeInspection = null;
            paintPanelButtons(dom);
            paintReport(dom);
            drawCanvas(dom);
        });
    });
}

function wireFreeInspection(dom: VlmDom) {
    const freeSubmit = document.getElementById(
        "free-query-submit",
    ) as HTMLButtonElement;

    const runInspection = async (query: string) => {
        if (!state.processor || !state.model || !state.currentFile) return;

        freeSubmit.disabled = true;
        dom.freeQueryInput.disabled = true;

        state.activePanel = "free-inspection";
        state.highlightedFindingId = null;

        updateReport(dom, "Running open inspection...");
        setStage(dom, `Inspection: ${query}`, 1);

        const live = startLiveMs(dom.evaluationTimer, 20);

        try {
            state.freeInspection = await runFreeInspection(
                state.currentFile,
                state.processor,
                state.model,
                query,
            );
            live.stop();
            paintReport(dom);
            drawCanvas(dom);
        } catch (error) {
            live.stop();
            updateReport(
                dom,
                `Open inspection error:\n${toErrorMessage(error)}`,
            );
        } finally {
            freeSubmit.disabled = false;
            dom.freeQueryInput.disabled = false;
        }
    };

    dom.freeQueryForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const query = dom.freeQueryInput.value.trim();
        if (!query) return;
        await runInspection(query);
    });

    dom.quickAlertButtons.forEach((button) => {
        button.addEventListener("click", async () => {
            const query = button.dataset.alert || "";
            if (!query) return;
            dom.freeQueryInput.value = query;
            await runInspection(query);
        });
    });
}

async function executeMasterScan(dom: VlmDom, file: File) {
    if (!state.processor || !state.model) return;

    state.currentFile = file;
    state.masterScan = null;
    state.freeInspection = null;
    state.activePanel = "description";
    state.highlightedFindingId = null;

    const bitmapPromise = createImageBitmap(file);

    dom.uploadPrompt.classList.add("hidden");
    dom.imageCanvas.classList.remove("hidden");

    updateReport(dom, "Running Master Scan. Please wait for the base pass...");
    setStage(dom, "Master Scan in progress", 0.16);

    dom.freeQueryInput.disabled = true;
    const freeSubmit = document.getElementById(
        "free-query-submit",
    ) as HTMLButtonElement;
    freeSubmit.disabled = true;

    const live = startLiveMs(dom.evaluationTimer, 20);

    try {
        state.imageBitmap = await bitmapPromise;
        drawCanvas(dom);

        state.masterScan = await runMasterScan(
            file,
            state.processor,
            state.model,
            (update) => {
                setStage(dom, update.stage, update.ratio);
            },
        );

        live.stop();
        setStage(dom, "Master Scan complete. Instant panels unlocked.", 1);

        dom.freeQueryInput.disabled = false;
        freeSubmit.disabled = false;

        paintPanelButtons(dom);
        paintReport(dom);
        drawCanvas(dom);
    } catch (error) {
        live.stop();
        setStage(dom, "Scan failed", 1);
        updateReport(dom, `Master Scan error:\n${toErrorMessage(error)}`);
    }
}

function paintPanelButtons(dom: VlmDom) {
    (
        Object.keys(dom.panelButtons) as Array<
            Exclude<ScanPanel, "free-inspection">
        >
    ).forEach((panel) => {
        const btn = dom.panelButtons[panel];
        if (state.activePanel === panel) {
            btn.className =
                "panel-btn px-2 py-2 rounded-lg text-[11px] font-bold bg-emerald-500 text-slate-950";
            return;
        }
        btn.className =
            "panel-btn px-2 py-2 rounded-lg text-[11px] font-bold text-slate-300 border border-slate-700";
    });
}

function paintReport(dom: VlmDom) {
    if (!state.masterScan && state.activePanel !== "free-inspection") {
        updateReport(dom, "Waiting for Master Scan...");
        return;
    }

    dom.reportContainer.innerHTML = "";

    if (state.activePanel === "free-inspection") {
        if (!state.freeInspection) {
            updateReport(dom, "No open-inspection result yet.");
            return;
        }

        appendSection(
            dom.reportContainer,
            "Open Inspection",
            state.freeInspection.lines,
        );
        return;
    }

    if (!state.masterScan) {
        updateReport(dom, "Waiting for Master Scan...");
        return;
    }

    if (state.activePanel === "description") {
        appendSection(
            dom.reportContainer,
            "General Description",
            state.masterScan.byTask["<DETAILED_CAPTION>"].lines,
        );
        return;
    }

    if (state.activePanel === "ocr") {
        appendSection(
            dom.reportContainer,
            "Text Reading (OCR)",
            state.masterScan.byTask["<OCR>"].lines,
        );
        return;
    }

    if (state.activePanel === "grid") {
        appendSection(
            dom.reportContainer,
            "Grid Scan",
            state.masterScan.gridSummary,
        );
        return;
    }

    appendSection(
        dom.reportContainer,
        "Detailed Detection",
        state.masterScan.byTask["<OD>"].lines,
    );

    const hint = document.createElement("p");
    hint.className = "text-xs text-slate-400 mt-4";
    hint.textContent =
        "Hover chips to highlight boxes in the image, or hover boxes to highlight chips.";
    dom.reportContainer.appendChild(hint);

    const tagsWrap = document.createElement("div");
    tagsWrap.className = "flex flex-wrap gap-2 mt-2";

    state.masterScan.findings.forEach((finding) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.dataset.findingId = finding.id;
        chip.className =
            "finding-chip px-2 py-1 rounded-full text-xs border border-slate-600 text-slate-200 hover:border-emerald-400";
        chip.textContent = `[${finding.label}]`;

        chip.addEventListener("mouseenter", () => {
            state.highlightedFindingId = finding.id;
            syncHighlightState(dom);
        });
        chip.addEventListener("mouseleave", () => {
            state.highlightedFindingId = null;
            syncHighlightState(dom);
        });

        tagsWrap.appendChild(chip);
    });

    dom.reportContainer.appendChild(tagsWrap);
    syncHighlightState(dom);
}

function appendSection(
    container: HTMLDivElement,
    title: string,
    lines: string[],
) {
    const titleEl = document.createElement("h3");
    titleEl.className =
        "text-xs uppercase tracking-wider text-emerald-300 font-bold mb-3";
    titleEl.textContent = title;

    const list = document.createElement("div");
    list.className = "space-y-2";

    lines.forEach((line) => {
        const item = document.createElement("p");
        item.className = "text-sm text-slate-200 leading-relaxed";
        item.textContent = line;
        list.appendChild(item);
    });

    container.append(titleEl, list);
}

function syncHighlightState(dom: VlmDom) {
    const chips = dom.reportContainer.querySelectorAll(
        ".finding-chip",
    ) as NodeListOf<HTMLButtonElement>;

    chips.forEach((chip) => {
        const isActive = chip.dataset.findingId === state.highlightedFindingId;
        chip.className = isActive
            ? "finding-chip px-2 py-1 rounded-full text-xs border border-emerald-300 text-emerald-200 bg-emerald-500/10"
            : "finding-chip px-2 py-1 rounded-full text-xs border border-slate-600 text-slate-200 hover:border-emerald-400";
    });

    drawCanvas(dom);
    renderOverlayBoxes(dom);
}

function drawCanvas(dom: VlmDom) {
    const canvas = dom.imageCanvas;
    const context = canvas.getContext("2d");
    if (!context || !state.imageBitmap) return;

    canvas.width = state.imageBitmap.width;
    canvas.height = state.imageBitmap.height;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(state.imageBitmap, 0, 0);

    const findings = getVisibleFindings();
    findings.forEach((finding) => {
        const [x1, y1, x2, y2] = finding.bbox;
        const active = finding.id === state.highlightedFindingId;

        context.strokeStyle = active
            ? "rgba(16, 185, 129, 1)"
            : "rgba(5, 150, 105, 0.9)";
        context.lineWidth = active ? 4 : 2;
        context.strokeRect(x1, y1, Math.max(2, x2 - x1), Math.max(2, y2 - y1));

        context.fillStyle = active
            ? "rgba(6, 95, 70, 0.95)"
            : "rgba(4, 120, 87, 0.9)";
        context.font = "600 14px ui-sans-serif";
        const text = finding.label;
        const textWidth = context.measureText(text).width;
        context.fillRect(x1, Math.max(0, y1 - 20), textWidth + 12, 20);
        context.fillStyle = "#f8fafc";
        context.fillText(text, x1 + 6, Math.max(14, y1 - 6));
    });

    renderOverlayBoxes(dom);
}

function renderOverlayBoxes(dom: VlmDom) {
    const layer = dom.overlayLayer;
    layer.innerHTML = "";

    if (!state.imageBitmap) {
        layer.classList.add("hidden");
        return;
    }

    const findings = getVisibleFindings();
    if (findings.length === 0) {
        layer.classList.add("hidden");
        return;
    }

    const canvasRect = dom.imageCanvas.getBoundingClientRect();
    if (canvasRect.width <= 0 || canvasRect.height <= 0) {
        layer.classList.add("hidden");
        return;
    }

    layer.classList.remove("hidden");
    layer.style.top = `${dom.imageCanvas.offsetTop}px`;
    layer.style.height = `${dom.imageCanvas.offsetHeight}px`;

    const imageWidth = state.imageBitmap.width;
    const imageHeight = state.imageBitmap.height;

    findings.forEach((finding) => {
        const [x1, y1, x2, y2] = finding.bbox;
        const left = clampPercent((Math.min(x1, x2) / imageWidth) * 100);
        const right = clampPercent((Math.max(x1, x2) / imageWidth) * 100);
        const top = clampPercent((Math.min(y1, y2) / imageHeight) * 100);
        const bottom = clampPercent((Math.max(y1, y2) / imageHeight) * 100);

        const box = document.createElement("div");
        const active = finding.id === state.highlightedFindingId;
        box.className = active
            ? "absolute pointer-events-auto border-2 border-emerald-400 bg-emerald-500/10 shadow-[0_0_0_2px_rgba(16,185,129,0.35)]"
            : "absolute pointer-events-auto border-2 border-emerald-600/95 bg-emerald-600/10";

        box.style.left = `${left}%`;
        box.style.top = `${top}%`;
        box.style.width = `${Math.max(0.8, right - left)}%`;
        box.style.height = `${Math.max(0.8, bottom - top)}%`;

        const label = document.createElement("div");
        label.className =
            "absolute -top-6 left-0 bg-emerald-700/95 text-emerald-50 text-[10px] font-bold px-1.5 py-0.5 rounded whitespace-nowrap";
        label.textContent = finding.label;
        box.appendChild(label);

        box.addEventListener("mouseenter", () => {
            state.highlightedFindingId = finding.id;
            syncHighlightState(dom);
        });

        box.addEventListener("mouseleave", () => {
            state.highlightedFindingId = null;
            syncHighlightState(dom);
        });

        layer.appendChild(box);
    });
}

function clampPercent(value: number): number {
    if (Number.isNaN(value) || !Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(100, value));
}

function getVisibleFindings(): GroundedFinding[] {
    if (!state.masterScan) return [];

    if (state.activePanel === "detection" || state.activePanel === "grid") {
        return state.masterScan.findings;
    }

    if (state.activePanel === "free-inspection" && state.freeInspection) {
        return state.freeInspection.findings;
    }

    return [];
}

function findHoveredFindingId(
    canvas: HTMLCanvasElement,
    event: MouseEvent,
    findings: GroundedFinding[],
): string | null {
    if (!findings.length) return null;

    const rect = canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (canvas.width / rect.width);
    const py = (event.clientY - rect.top) * (canvas.height / rect.height);

    for (let i = findings.length - 1; i >= 0; i -= 1) {
        const finding = findings[i];
        const [x1, y1, x2, y2] = finding.bbox;
        if (px >= x1 && px <= x2 && py >= y1 && py <= y2) {
            return finding.id;
        }
    }

    return null;
}

function setStage(dom: VlmDom, message: string, ratio: number) {
    dom.scanStage.textContent = message;
    const bounded = Math.max(0, Math.min(1, ratio));
    dom.scanProgressBar.style.width = `${Math.round(bounded * 100)}%`;
}

function updateReport(dom: VlmDom, text: string) {
    dom.reportContainer.textContent = text;
}

function toErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    return String(error);
}
