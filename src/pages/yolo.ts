import { hasGPU } from "../lib/has-gpu";
import { startLiveMs } from "../lib/live-ms";
import { Timer } from "../lib/timer";
import {
    detectWithYolo,
    getYoloDetector,
    type YoloDetection,
} from "../services/yolo-service";

interface YoloState {
    useGpu: boolean;
    detectorReady: boolean;
    currentFile: File | null;
    imageBitmap: ImageBitmap | null;
    detections: YoloDetection[];
    cameraStream: MediaStream | null;
}

const state: YoloState = {
    useGpu: false,
    detectorReady: false,
    currentFile: null,
    imageBitmap: null,
    detections: [],
    cameraStream: null,
};

export async function render(app: HTMLElement) {
    document.title = "YOLOv8n Object Detection";

    app.innerHTML = `
    <div class="min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-300 selection:text-slate-950">
      <div class="mx-auto max-w-6xl px-4 py-6 md:px-8 md:py-10">
        <header class="mb-5 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 md:p-6">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div class="flex items-center gap-2">
                <span class="rounded-full border border-emerald-300/40 bg-emerald-300/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-200">Edge YOLO</span>
                <a href="${import.meta.env.BASE_URL}" class="text-xs text-slate-400 underline decoration-slate-600 hover:text-slate-200">back to catalog</a>
              </div>
              <h1 class="mt-2 text-2xl font-black tracking-tight text-white md:text-4xl">YOLOv8n Object Detection (Local)</h1>
              <p class="mt-2 max-w-3xl text-sm text-slate-300">Runs the lightest YOLOv8 model (YOLOv8n Nano) in-browser and draws bounding boxes with confidence scores.</p>
            </div>
            <div class="rounded-xl border border-slate-700/70 bg-slate-950/60 p-3 text-xs font-mono">
              <div class="flex items-center justify-between gap-4"><span class="text-slate-400">Runtime</span><span id="hardware-badge" class="text-emerald-300">Detecting...</span></div>
              <div class="mt-2 flex items-center justify-between gap-4"><span class="text-slate-500">Model load</span><span id="load-ms" class="font-bold text-emerald-300">-- ms</span></div>
              <div class="mt-1 flex items-center justify-between gap-4"><span class="text-slate-500">Detection</span><span id="detect-ms" class="font-bold text-emerald-300">-- ms</span></div>
            </div>
          </div>
        </header>

        <main class="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_320px]">
          <section class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <label id="drop-area" class="block cursor-pointer rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/60 p-4 transition-colors hover:border-emerald-400/70">
              <input id="file-input" type="file" accept="image/*" class="hidden" />
              <div id="upload-prompt" class="py-12 text-center">
                <div class="text-5xl">🧭</div>
                <p class="mt-2 text-sm font-bold text-emerald-300">Drop an image to run YOLOv8n</p>
                <p class="mt-1 text-xs text-slate-500">Uses the lightest YOLOv8 model for faster local inference.</p>
              </div>
              <video id="camera-video" autoplay playsinline muted class="hidden w-full rounded-xl border border-slate-700/60 bg-slate-950"></video>
              <canvas id="image-canvas" class="hidden w-full rounded-xl border border-slate-700/60 bg-slate-950"></canvas>
            </label>

            <div class="mt-3 flex flex-wrap items-center gap-2">
              <label for="threshold-input" class="text-xs font-bold uppercase tracking-wider text-slate-300">Confidence</label>
              <input id="threshold-input" type="number" min="0.05" max="0.95" step="0.05" value="0.25" class="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200" />
              <button id="detect-btn" disabled class="rounded-lg bg-emerald-300 px-4 py-2 text-xs font-black uppercase tracking-wider text-slate-950 disabled:cursor-not-allowed disabled:opacity-45">Detect objects</button>
            </div>

            <div class="mt-2 flex flex-wrap items-center gap-2">
              <button id="camera-start-btn" class="rounded-lg border border-emerald-600/60 px-3 py-2 text-xs font-bold uppercase tracking-wider text-emerald-200 hover:border-emerald-400">Start camera</button>
              <button id="camera-capture-btn" disabled class="rounded-lg bg-emerald-500 px-3 py-2 text-xs font-bold uppercase tracking-wider text-slate-950 disabled:cursor-not-allowed disabled:opacity-45">Capture frame</button>
              <button id="camera-stop-btn" disabled class="rounded-lg border border-slate-700 px-3 py-2 text-xs font-bold uppercase tracking-wider text-slate-300 disabled:cursor-not-allowed disabled:opacity-45">Stop camera</button>
              <span id="camera-status" class="text-[11px] text-slate-500">Camera idle</span>
            </div>
          </section>

          <aside class="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
            <h2 class="text-sm font-black uppercase tracking-wider text-white">Detections</h2>
            <div id="detection-list" class="mt-3 space-y-2 text-sm text-slate-300">
              <p class="text-slate-500">No detections yet.</p>
            </div>
          </aside>
        </main>
      </div>
    </div>
    `;

    await boot();
}

async function boot() {
    const hardwareBadge = document.getElementById(
        "hardware-badge",
    ) as HTMLSpanElement;
    const loadMs = document.getElementById("load-ms") as HTMLSpanElement;
    const detectMs = document.getElementById("detect-ms") as HTMLSpanElement;
    const fileInput = document.getElementById("file-input") as HTMLInputElement;
    const dropArea = document.getElementById("drop-area") as HTMLLabelElement;
    const uploadPrompt = document.getElementById(
        "upload-prompt",
    ) as HTMLDivElement;
    const cameraVideo = document.getElementById(
        "camera-video",
    ) as HTMLVideoElement;
    const imageCanvas = document.getElementById(
        "image-canvas",
    ) as HTMLCanvasElement;
    const thresholdInput = document.getElementById(
        "threshold-input",
    ) as HTMLInputElement;
    const detectBtn = document.getElementById(
        "detect-btn",
    ) as HTMLButtonElement;
    const cameraStartBtn = document.getElementById(
        "camera-start-btn",
    ) as HTMLButtonElement;
    const cameraCaptureBtn = document.getElementById(
        "camera-capture-btn",
    ) as HTMLButtonElement;
    const cameraStopBtn = document.getElementById(
        "camera-stop-btn",
    ) as HTMLButtonElement;
    const cameraStatus = document.getElementById(
        "camera-status",
    ) as HTMLSpanElement;
    const detectionList = document.getElementById(
        "detection-list",
    ) as HTMLDivElement;

    state.useGpu = await hasGPU();
    hardwareBadge.innerText = state.useGpu ? "WebGPU" : "WASM";

    const liveLoad = startLiveMs(loadMs, 20);
    const [detector, modelMs] = await Timer.wrap(() =>
        getYoloDetector(state.useGpu),
    )();
    liveLoad.stop(modelMs);
    state.detectorReady = true;

    const onFile = async (file: File) => {
        if (!file.type.startsWith("image/")) return;

        state.currentFile = file;
        state.imageBitmap = await createImageBitmap(file);
        state.detections = [];

        uploadPrompt.classList.add("hidden");
        cameraVideo.classList.add("hidden");
        imageCanvas.classList.remove("hidden");
        detectBtn.disabled = !state.detectorReady;

        drawPreview(imageCanvas, state.imageBitmap);
        detectionList.innerHTML = `<p class="text-slate-500">Image loaded. Click Detect objects.</p>`;
    };

    fileInput.addEventListener("change", async (event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (!file) return;
        await onFile(file);
    });

    dropArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropArea.classList.add("border-emerald-400");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("border-emerald-400");
    });

    dropArea.addEventListener("drop", async (event) => {
        event.preventDefault();
        dropArea.classList.remove("border-emerald-400");

        const file = event.dataTransfer?.files?.[0];
        if (!file) return;

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        await onFile(file);
    });

    detectBtn.addEventListener("click", async () => {
        if (!state.currentFile || !state.imageBitmap || !state.detectorReady)
            return;

        detectBtn.disabled = true;
        detectionList.innerHTML = `<p class="text-slate-400">Detecting objects...</p>`;

        const threshold = clampThreshold(Number(thresholdInput.value));
        const liveDetect = startLiveMs(detectMs, 20);

        try {
            const [detections, elapsedMs] = await Timer.wrap(() =>
                detectWithYolo(state.currentFile!, detector, threshold),
            )();
            liveDetect.stop(elapsedMs);

            state.detections = detections;
            drawDetections(imageCanvas, state.imageBitmap, detections);
            renderDetectionsList(detectionList, detections);
        } catch (error) {
            liveDetect.stop();
            detectionList.innerHTML = `<p class="text-emerald-300">Detection failed: ${escapeHtml(toErrorMessage(error))}</p>`;
        } finally {
            detectBtn.disabled = false;
        }
    });

    if (!navigator.mediaDevices?.getUserMedia) {
        cameraStartBtn.disabled = true;
        cameraStatus.innerText = "Camera API unavailable";
        return;
    }

    const stopCamera = () => {
        stopMediaStream(state.cameraStream);
        state.cameraStream = null;

        cameraVideo.pause();
        cameraVideo.srcObject = null;
        cameraVideo.classList.add("hidden");

        cameraCaptureBtn.disabled = true;
        cameraStopBtn.disabled = true;
        cameraStartBtn.disabled = false;
        cameraStatus.innerText = "Camera stopped";
    };

    cameraStartBtn.addEventListener("click", async () => {
        if (state.cameraStream) return;

        cameraStatus.innerText = "Starting camera...";
        try {
            let stream: MediaStream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" },
                    audio: false,
                });
            } catch {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false,
                });
            }

            state.cameraStream = stream;
            cameraVideo.srcObject = stream;
            await cameraVideo.play();

            uploadPrompt.classList.add("hidden");
            imageCanvas.classList.add("hidden");
            cameraVideo.classList.remove("hidden");

            cameraCaptureBtn.disabled = false;
            cameraStopBtn.disabled = false;
            cameraStartBtn.disabled = true;
            cameraStatus.innerText = "Camera active";
        } catch (error) {
            cameraStatus.innerText = `Camera error: ${toErrorMessage(error)}`;
        }
    });

    cameraCaptureBtn.addEventListener("click", async () => {
        if (!state.cameraStream) return;

        cameraCaptureBtn.disabled = true;
        cameraStatus.innerText = "Capturing frame...";

        try {
            const cameraFile = await captureVideoFrameToFile(
                cameraVideo,
                "yolo-camera",
            );
            await onFile(cameraFile);
            cameraStatus.innerText = "Frame captured";
        } catch (error) {
            cameraStatus.innerText = `Capture failed: ${toErrorMessage(error)}`;
        } finally {
            cameraCaptureBtn.disabled = false;
        }
    });

    cameraStopBtn.addEventListener("click", () => {
        stopCamera();
    });

    window.addEventListener("beforeunload", () => {
        stopMediaStream(state.cameraStream);
    });
}

function drawPreview(canvas: HTMLCanvasElement, bitmap: ImageBitmap) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0);
}

function drawDetections(
    canvas: HTMLCanvasElement,
    bitmap: ImageBitmap,
    detections: YoloDetection[],
) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawPreview(canvas, bitmap);

    for (const detection of detections) {
        const [x1Raw, y1Raw, x2Raw, y2Raw] = detection.bbox;
        const [x1, y1, x2, y2] = denormalizeBox(
            [x1Raw, y1Raw, x2Raw, y2Raw],
            canvas.width,
            canvas.height,
        );

        const width = Math.max(2, x2 - x1);
        const height = Math.max(2, y2 - y1);

        ctx.strokeStyle = "rgba(16, 185, 129, 1)";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        const tag = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;
        ctx.font = "600 13px ui-sans-serif";
        const textWidth = ctx.measureText(tag).width;
        ctx.fillStyle = "rgba(6, 95, 70, 0.92)";
        ctx.fillRect(x1, Math.max(0, y1 - 20), textWidth + 10, 18);
        ctx.fillStyle = "#ecfdf5";
        ctx.fillText(tag, x1 + 5, Math.max(13, y1 - 7));
    }
}

function denormalizeBox(
    box: [number, number, number, number],
    width: number,
    height: number,
): [number, number, number, number] {
    const [x1, y1, x2, y2] = box;
    const looksNormalized =
        Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) <= 1.5;

    if (!looksNormalized) {
        return [x1, y1, x2, y2];
    }

    return [x1 * width, y1 * height, x2 * width, y2 * height];
}

function renderDetectionsList(
    container: HTMLDivElement,
    detections: YoloDetection[],
) {
    if (!detections.length) {
        container.innerHTML = `<p class="text-slate-500">No objects found for the selected threshold.</p>`;
        return;
    }

    container.innerHTML = detections
        .slice(0, 20)
        .map(
            (detection, index) => `
            <article class="rounded-lg border border-slate-800 bg-slate-950/70 p-2">
              <div class="mb-1 flex items-center justify-between">
                <span class="text-[10px] font-bold uppercase tracking-wider text-emerald-300">#${index + 1} ${escapeHtml(detection.label)}</span>
                <span class="text-[10px] font-mono text-slate-400">${(detection.score * 100).toFixed(1)}%</span>
              </div>
              <p class="text-[11px] text-slate-400">bbox [${detection.bbox.map((n) => Math.round(n)).join(", ")}]</p>
            </article>
          `,
        )
        .join("");
}

function clampThreshold(value: number): number {
    if (!Number.isFinite(value)) return 0.25;
    return Math.max(0.05, Math.min(0.95, value));
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

async function captureVideoFrameToFile(
    video: HTMLVideoElement,
    baseName: string,
): Promise<File> {
    if (!video.videoWidth || !video.videoHeight) {
        throw new Error("Camera has no frame available yet.");
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) {
        throw new Error("Could not prepare frame capture context.");
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
            (value) => {
                if (!value) {
                    reject(new Error("Could not encode captured frame."));
                    return;
                }
                resolve(value);
            },
            "image/png",
            0.95,
        );
    });

    return new File([blob], `${baseName}-${Date.now()}.png`, {
        type: "image/png",
    });
}

function stopMediaStream(stream: MediaStream | null) {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
}
