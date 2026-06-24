import {
    TextClassificationPipeline,
    pipeline,
    env,
} from "@huggingface/transformers";

export async function getEvaluation(
    text: string,
    pipe: TextClassificationPipeline,
): Promise<{ label: string; score: number }> {
    const out = await pipe(text);

    return out[0];
}

if (env.backends.onnx.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.simd = true;
}

export async function getEvaluationPipeline(
    useGpu: boolean = false,
): Promise<TextClassificationPipeline> {
    const pipe = await pipeline(
        "sentiment-analysis",
        "Xenova/bert-base-multilingual-uncased-sentiment",
        { dtype: "q8", device: useGpu ? "webgpu" : "wasm" },
    );

    return pipe;
}
