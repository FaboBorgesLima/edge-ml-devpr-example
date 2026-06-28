import { env, pipeline } from "@huggingface/transformers";

export interface YoloDetection {
    label: string;
    score: number;
    bbox: [number, number, number, number];
}

type ObjectDetectionPipeline = (
    input: string,
    options?: unknown,
) => Promise<unknown>;

const YOLOV8_NANO_MODEL = "onnx-community/yolov8n";

let webgpuDetector: ObjectDetectionPipeline | null = null;
let wasmDetector: ObjectDetectionPipeline | null = null;

if (env.backends.onnx.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.simd = true;
}

env.allowLocalModels = false;

export async function getYoloDetector(
    useGpu: boolean,
): Promise<ObjectDetectionPipeline> {
    if (useGpu && webgpuDetector) return webgpuDetector;
    if (!useGpu && wasmDetector) return wasmDetector;

    const detector = (await pipeline("object-detection", YOLOV8_NANO_MODEL, {
        device: useGpu ? "webgpu" : "wasm",
    })) as unknown as ObjectDetectionPipeline;

    if (useGpu) {
        webgpuDetector = detector;
    } else {
        wasmDetector = detector;
    }

    return detector;
}

export async function detectWithYolo(
    imageFile: File,
    detector: ObjectDetectionPipeline,
    threshold: number = 0.25,
): Promise<YoloDetection[]> {
    const imageUrl = URL.createObjectURL(imageFile);

    try {
        const rawOutput = await detector(imageUrl, { threshold });
        return normalizeDetections(rawOutput);
    } finally {
        URL.revokeObjectURL(imageUrl);
    }
}

function normalizeDetections(rawOutput: unknown): YoloDetection[] {
    if (!Array.isArray(rawOutput)) return [];

    const parsed: YoloDetection[] = [];

    for (const item of rawOutput) {
        if (typeof item !== "object" || item === null) continue;

        const record = item as {
            label?: unknown;
            score?: unknown;
            box?: {
                xmin?: unknown;
                ymin?: unknown;
                xmax?: unknown;
                ymax?: unknown;
            };
            bbox?: unknown;
        };

        const label = String(record.label ?? "object").trim();
        const score = toNumber(record.score, 0);

        let x1 = 0;
        let y1 = 0;
        let x2 = 0;
        let y2 = 0;

        if (record.box) {
            x1 = toNumber(record.box.xmin, 0);
            y1 = toNumber(record.box.ymin, 0);
            x2 = toNumber(record.box.xmax, 0);
            y2 = toNumber(record.box.ymax, 0);
        } else if (Array.isArray(record.bbox) && record.bbox.length >= 4) {
            x1 = toNumber(record.bbox[0], 0);
            y1 = toNumber(record.bbox[1], 0);
            x2 = toNumber(record.bbox[2], 0);
            y2 = toNumber(record.bbox[3], 0);
        }

        parsed.push({
            label,
            score,
            bbox: [
                Math.min(x1, x2),
                Math.min(y1, y2),
                Math.max(x1, x2),
                Math.max(y1, y2),
            ],
        });
    }

    return parsed.sort((a, b) => b.score - a.score);
}

function toNumber(value: unknown, fallback: number): number {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}
