import {
    env,
    Florence2ForConditionalGeneration,
    AutoProcessor,
    Processor,
    PreTrainedModel,
    Tensor,
    load_image,
} from "@huggingface/transformers";

// --- PREPARAÇÃO DE MEMÓRIA PARA MULTI-NAVEGADORES ---
// Desativa recursos experimentais que quebram o Firefox e Safari
if (env.backends.onnx.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.simd = true;
}
// Evita que o ONNX tente usar caminhos locais inexistentes no servidor de desenvolvimento
env.allowLocalModels = false;

export type SupportedTasks = "<OCR>" | "<DETAILED_CAPTION>" | "<OD>";

export type ScanPanel =
    | "description"
    | "ocr"
    | "detection"
    | "grid"
    | "free-inspection";

export interface GroundedFinding {
    id: string;
    label: string;
    bbox: [number, number, number, number];
}

export interface TaskScanResult {
    task: SupportedTasks;
    text: string;
    lines: string[];
    findings: GroundedFinding[];
}

export interface MasterScanResult {
    byTask: Record<SupportedTasks, TaskScanResult>;
    findings: GroundedFinding[];
    gridSummary: string[];
}

export interface ScanProgressUpdate {
    stage: string;
    ratio: number;
}

interface ImageSize {
    width: number;
    height: number;
}

type FlorenceTaskPrompt = SupportedTasks | "<OPEN_VOCABULARY_DETECTION>";

/**
 * VLM - Visual Language Model
 * @description This function loads the Florence-2-base model and its corresponding processor for visual language tasks.
 * @description What is a Visual Language Model (VLM)? A VLM is a type of machine learning model that can understand and generate text based on visual inputs, such as images. It combines computer vision and natural language processing to perform tasks like image captioning, object detection, and optical character recognition (OCR).
 * @description What is the difference between Florence-2-base and other VLMs? Florence-2-base is a specific implementation of a VLM that has been trained on a large dataset of images and text. It is designed to perform well on a variety of visual language tasks, making it versatile for different applications.
 * @description Why it is not that good for Edge devices? Florence-2-base is a large model that requires significant computational resources to run efficiently. Edge devices, such as smartphones and IoT devices, often have limited processing power and memory, which can lead to slower performance and higher latency when using large models like Florence-2-base.
 * @returns
 */
export async function getVlm(
    useGpu: boolean = false,
): Promise<[Processor, PreTrainedModel]> {
    const model_id = "onnx-community/Florence-2-base";
    const model = await Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        { dtype: "fp32", device: useGpu ? "webgpu" : "wasm" },
    );
    const processor = await AutoProcessor.from_pretrained(model_id);

    return [processor, model];
}

async function runTask(
    processedImage: Awaited<ReturnType<typeof load_image>>,
    processor: Processor,
    model: PreTrainedModel,
    taskPrompt: FlorenceTaskPrompt | string,
    outputTask: SupportedTasks,
): Promise<TaskScanResult> {
    const processorWithPrompts = processor as Processor & {
        construct_prompts?: (task: string) => string | string[];
    };

    const prompts = processorWithPrompts.construct_prompts
        ? processorWithPrompts.construct_prompts(taskPrompt)
        : taskPrompt;

    const inputs = await processor(processedImage, prompts as any);
    const generatedIds = (await model.generate({
        ...inputs,
        max_new_tokens: 180,
    })) as Tensor;

    const generatedText = processor.batch_decode(generatedIds, {
        skip_special_tokens: false,
    })[0];

    const postProcessor = processor as Processor & {
        post_process_generation?: (
            text: string,
            task: SupportedTasks,
            image: Awaited<ReturnType<typeof load_image>>,
        ) => unknown;
    };
    console.log(`Task: ${taskPrompt}, Output Task: ${outputTask}`);
    console.log(`Generated text:`, generatedText);

    const postProcessed = postProcessor.post_process_generation
        ? postProcessor.post_process_generation(
              generatedText,
              outputTask,
              processedImage,
          )
        : { [outputTask]: generatedText };
    console.log(`Post-processed output for task ${outputTask}:`, postProcessed);

    const imageSize = getImageSize(processedImage);

    return normalizeTaskResult(
        outputTask,
        postProcessed,
        generatedText,
        imageSize,
    );
}

function normalizeTaskResult(
    task: SupportedTasks,
    payload: unknown,
    fallbackText: string,
    imageSize?: ImageSize,
): TaskScanResult {
    const root =
        typeof payload === "object" && payload !== null
            ? (payload as Record<string, unknown>)
            : {};
    const taskPayload = root[task] ?? payload;

    let findings = extractFindings(taskPayload);
    const lines = extractLines(taskPayload, task);

    // PT: Se o pos-processamento falhar (NaN/null), extraimos <loc_*> direto do texto gerado.
    // EN: If post-processing fails (NaN/null), recover detections directly from <loc_*> tokens.
    if (task === "<OD>" && findings.length === 0) {
        const recovered = parseFlorenceLocDetections(fallbackText, imageSize);
        if (recovered.length > 0) {
            findings = recovered;
        }
    }

    let normalizedLines = lines.length
        ? lines
        : [fallbackText.trim() || "No textual output for this task."];

    if (task === "<OD>" && findings.length > 0) {
        normalizedLines = findings.map(
            (finding, index) =>
                `${index + 1}. ${finding.label} -> bbox [${finding.bbox
                    .map((value) => Math.round(value))
                    .join(", ")}]`,
        );
    }

    return {
        task,
        text: normalizedLines.join("\n"),
        lines: normalizedLines,
        findings,
    };
}

function getImageSize(image: unknown): ImageSize | undefined {
    if (typeof image !== "object" || image === null) {
        return undefined;
    }

    const candidate = image as { width?: unknown; height?: unknown };
    const width = toFiniteNumber(candidate.width);
    const height = toFiniteNumber(candidate.height);

    if (width === null || height === null || width <= 0 || height <= 0) {
        return undefined;
    }

    return { width, height };
}

function parseFlorenceLocDetections(
    generatedText: string,
    imageSize?: ImageSize,
): GroundedFinding[] {
    if (!generatedText.includes("<loc_")) {
        return [];
    }

    const detections: GroundedFinding[] = [];
    const regex = /([^<]*?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/g;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(generatedText)) !== null) {
        const rawLabel = sanitizeFlorenceLabel(match[1]);
        const label = rawLabel || `Object ${detections.length + 1}`;

        const x1Token = Number(match[2]);
        const y1Token = Number(match[3]);
        const x2Token = Number(match[4]);
        const y2Token = Number(match[5]);

        if ([x1Token, y1Token, x2Token, y2Token].some((v) => Number.isNaN(v))) {
            continue;
        }

        let x1 = x1Token;
        let y1 = y1Token;
        let x2 = x2Token;
        let y2 = y2Token;

        if (imageSize) {
            x1 = (x1Token / 999) * imageSize.width;
            y1 = (y1Token / 999) * imageSize.height;
            x2 = (x2Token / 999) * imageSize.width;
            y2 = (y2Token / 999) * imageSize.height;
        }

        const left = Math.min(x1, x2);
        const right = Math.max(x1, x2);
        const top = Math.min(y1, y2);
        const bottom = Math.max(y1, y2);

        detections.push({
            id: `finding-fallback-${detections.length}`,
            label: label.replace("s>", "").trim(),
            bbox: [left, top, right, bottom],
        });
    }

    return detections;
}

function sanitizeFlorenceLabel(raw: string): string {
    return raw
        .replace(/<\/?s>/g, " ")
        .replace(/<pad>/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function extractLines(payload: unknown, task?: SupportedTasks): string[] {
    if (typeof payload === "string") {
        return payload
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean);
    }

    if (Array.isArray(payload)) {
        return payload.map((item) => String(item)).filter(Boolean);
    }

    if (typeof payload !== "object" || payload === null) {
        return [];
    }

    const record = payload as Record<string, unknown>;
    const lineCandidates: string[] = [];

    if (typeof record.text === "string") lineCandidates.push(record.text);
    if (typeof record.caption === "string") lineCandidates.push(record.caption);
    if (Array.isArray(record.captions)) {
        lineCandidates.push(...record.captions.map((x) => String(x)));
    }
    if (Array.isArray(record.labels)) {
        const labels = record.labels.map((x) => String(x));
        if (task === "<OD>" && Array.isArray(record.bboxes)) {
            const objectLines = buildDetectionLines(labels, record.bboxes);
            if (objectLines.length > 0) {
                lineCandidates.push(...objectLines);
            } else if (labels.length > 0) {
                lineCandidates.push(
                    ...labels.map(
                        (label, index) =>
                            `${index + 1}. ${label} -> bbox indisponivel (modelo retornou coordenadas invalidas)`,
                    ),
                );
            }
        } else {
            lineCandidates.push(`Objects: ${labels.join(", ")}`);
        }
    }

    if (lineCandidates.length > 0) {
        return lineCandidates.map((line) => line.trim()).filter(Boolean);
    }

    return [JSON.stringify(record, null, 2)];
}

function buildDetectionLines(labels: string[], bboxes: unknown[]): string[] {
    const lines: string[] = [];

    for (let i = 0; i < bboxes.length; i += 1) {
        const box = bboxes[i];
        if (!Array.isArray(box) || box.length < 4) continue;

        const parsed = parseBBox(box);
        if (!parsed) continue;

        const [x1, y1, x2, y2] = parsed;

        const label = labels[i] || `Object ${i + 1}`;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        lines.push(
            `${i + 1}. ${label} -> bbox [${Math.round(x1)}, ${Math.round(y1)}, ${Math.round(x2)}, ${Math.round(y2)}], size ${Math.round(width)}x${Math.round(height)}`,
        );
    }

    return lines;
}

function toFiniteNumber(value: unknown): number | null {
    if (value === null || value === undefined || value === "") {
        return null;
    }

    const numeric =
        typeof value === "number" ? value : Number(String(value).trim());

    return Number.isFinite(numeric) ? numeric : null;
}

function parseBBox(box: unknown[]): [number, number, number, number] | null {
    if (box.length < 4) return null;

    const x1 = toFiniteNumber(box[0]);
    const y1 = toFiniteNumber(box[1]);
    const x2 = toFiniteNumber(box[2]);
    const y2 = toFiniteNumber(box[3]);

    if (x1 === null || y1 === null || x2 === null || y2 === null) {
        return null;
    }

    return [x1, y1, x2, y2];
}

function extractFindings(payload: unknown): GroundedFinding[] {
    if (typeof payload !== "object" || payload === null) {
        return [];
    }

    const record = payload as Record<string, unknown>;
    const labels = Array.isArray(record.labels)
        ? record.labels.map((x) => String(x))
        : [];
    const bboxes = Array.isArray(record.bboxes) ? record.bboxes : [];

    const findings: GroundedFinding[] = [];

    for (let i = 0; i < bboxes.length; i += 1) {
        const box = bboxes[i];
        if (!Array.isArray(box) || box.length < 4) {
            continue;
        }

        const parsed = parseBBox(box);
        if (!parsed) {
            continue;
        }

        const [x1, y1, x2, y2] = parsed;

        findings.push({
            id: `finding-${i}`,
            label: labels[i] || `Object ${i + 1}`,
            bbox: [x1, y1, x2, y2],
        });
    }

    return findings;
}

function buildGridSummary(findings: GroundedFinding[]): string[] {
    if (findings.length === 0) {
        return ["No grounded detections available for grid summary."];
    }

    const buckets = {
        "Top Left": 0,
        "Top Right": 0,
        "Bottom Left": 0,
        "Bottom Right": 0,
    };

    findings.forEach((finding) => {
        const [x1, y1, x2, y2] = finding.bbox;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const horizontal = cx < 500 ? "Left" : "Right";
        const vertical = cy < 500 ? "Top" : "Bottom";
        const key = `${vertical} ${horizontal}` as keyof typeof buckets;
        buckets[key] += 1;
    });

    return Object.entries(buckets).map(
        ([name, count]) => `${name}: ${count} localized object(s)`,
    );
}

export async function runMasterScan(
    image: File,
    processor: Processor,
    model: PreTrainedModel,
    onProgress?: (update: ScanProgressUpdate) => void,
): Promise<MasterScanResult> {
    onProgress?.({ stage: "Loading image tensor", ratio: 0.06 });
    const processedImage = await load_image(image);

    const tasks: Array<{ task: SupportedTasks; stage: string }> = [
        { task: "<DETAILED_CAPTION>", stage: "Generating global description" },
        { task: "<OCR>", stage: "Reading text regions (OCR)" },
        { task: "<OD>", stage: "Grounding objects with coordinates" },
    ];

    const byTask = {} as Record<SupportedTasks, TaskScanResult>;

    for (let i = 0; i < tasks.length; i += 1) {
        const current = tasks[i];
        const ratio = 0.1 + (i / tasks.length) * 0.85;
        onProgress?.({ stage: current.stage, ratio });
        byTask[current.task] = await runTask(
            processedImage,
            processor,
            model,
            current.task,
            current.task,
        );
    }

    const findings = byTask["<OD>"].findings;
    const gridSummary = buildGridSummary(findings);

    onProgress?.({ stage: "Master scan ready", ratio: 1 });

    return {
        byTask,
        findings,
        gridSummary,
    };
}

export async function runFreeInspection(
    image: File,
    processor: Processor,
    model: PreTrainedModel,
    query: string,
): Promise<TaskScanResult> {
    const processedImage = await load_image(image);
    const prompt = `<OPEN_VOCABULARY_DETECTION> ${query}`;
    const result = await runTask(
        processedImage,
        processor,
        model,
        prompt,
        "<OD>",
    );

    const trimmedQuery = query.trim().toLowerCase();
    const filteredFindings = trimmedQuery
        ? result.findings.filter((finding) =>
              finding.label.toLowerCase().includes(trimmedQuery),
          )
        : result.findings;

    const findings =
        filteredFindings.length > 0 ? filteredFindings : result.findings;

    const lines: string[] = [];
    lines.push(`Query: ${query}`);
    if (findings.length > 0) {
        lines.push(
            ...findings.map(
                (finding, index) =>
                    `${index + 1}. ${finding.label} -> [${finding.bbox
                        .map((value) => Math.round(value))
                        .join(", ")}]`,
            ),
        );
    } else {
        lines.push(...result.lines);
    }

    return {
        task: "<OD>",
        text: lines.join("\n"),
        lines,
        findings,
    };
}

export async function processMultimodalImage(
    image: File,
    processor: Processor,
    model: PreTrainedModel,
    task: SupportedTasks,
): Promise<string> {
    const processedImage = await load_image(image);
    const result = await runTask(processedImage, processor, model, task, task);
    return result.text;
}
