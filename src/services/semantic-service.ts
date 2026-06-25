import { pipeline } from "@huggingface/transformers";

export interface SemanticHit {
    text: string;
    score: number;
    index: number;
}

type FeatureExtractor = any;

let webgpuExtractor: FeatureExtractor | null = null;
let wasmExtractor: FeatureExtractor | null = null;

async function getExtractor(useGpu: boolean): Promise<FeatureExtractor> {
    if (useGpu && webgpuExtractor) return webgpuExtractor;
    if (!useGpu && wasmExtractor) return wasmExtractor;

    const extractor = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2",
        {
            device: useGpu ? "webgpu" : "wasm",
        },
    );

    if (useGpu) {
        webgpuExtractor = extractor;
    } else {
        wasmExtractor = extractor;
    }

    return extractor;
}

export async function warmupSemanticModel(useGpu: boolean): Promise<void> {
    const extractor = await getExtractor(useGpu);
    await extractor("warmup", { pooling: "mean", normalize: true });
}

async function embedText(text: string, useGpu: boolean): Promise<number[]> {
    const extractor = await getExtractor(useGpu);
    const embedding = await extractor(text, {
        pooling: "mean",
        normalize: true,
    });
    return toVector(embedding);
}

function toVector(value: unknown): number[] {
    if (!value) return [];

    if (typeof value === "object" && value !== null && "data" in value) {
        const data = (value as { data?: ArrayLike<number> }).data;
        if (data && typeof data.length === "number") {
            return Array.from(data);
        }
    }

    if (Array.isArray(value)) {
        if (value.length === 0) return [];
        if (Array.isArray(value[0])) {
            return (value[0] as ArrayLike<number>).length !== undefined
                ? Array.from(value[0] as ArrayLike<number>)
                : [];
        }
        return Array.from(value as ArrayLike<number>);
    }

    return [];
}

function cosineSimilarity(a: number[], b: number[]): number {
    if (!a.length || !b.length || a.length !== b.length) return 0;

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i += 1) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function rankBySemanticSimilarity(
    query: string,
    documents: string[],
    useGpu: boolean,
    limit: number = 5,
): Promise<SemanticHit[]> {
    const cleanDocs = documents
        .map((doc) => doc.trim())
        .filter((doc) => doc.length > 0);

    if (!query.trim() || cleanDocs.length === 0) return [];

    const queryEmbedding = await embedText(query, useGpu);

    const hits: SemanticHit[] = [];
    for (let i = 0; i < cleanDocs.length; i += 1) {
        const docEmbedding = await embedText(cleanDocs[i], useGpu);
        hits.push({
            text: cleanDocs[i],
            score: cosineSimilarity(queryEmbedding, docEmbedding),
            index: i,
        });
    }

    return hits.sort((a, b) => b.score - a.score).slice(0, limit);
}
