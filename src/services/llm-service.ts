import {
    pipeline,
    TextGenerationPipeline,
    type Message,
    type PretrainedModelOptions,
} from "@huggingface/transformers";

export interface AvailableModel {
    key: string;
    name: string;
    webgpuDType: PretrainedModelOptions["dtype"];
    wasmDType: PretrainedModelOptions["dtype"];
}

export const AVAILABLE_MODELS: readonly AvailableModel[] = [
    {
        key: "HuggingFaceTB/SmolLM2-360M-Instruct",
        name: "SmolLM2 360M (Recommended)",
        webgpuDType: "q4f16",
        wasmDType: "int8",
    },
    {
        key: "HuggingFaceTB/SmolLM2-135M-Instruct",
        name: "SmolLM2 135M (Light)",
        webgpuDType: "q4f16",
        wasmDType: "int8",
    },
    {
        key: "onnx-community/gemma-3-270m-it-ONNX",
        name: "Gemma 3 270M (Heavy)",
        webgpuDType: "q4f16",
        wasmDType: "fp16",
    },
];

export async function getGenerator(
    useGpu: boolean,
    availableModel: AvailableModel,
) {
    return await pipeline("text-generation", availableModel.key, {
        device: useGpu ? "webgpu" : "wasm",
        dtype: useGpu ? availableModel.webgpuDType : availableModel.wasmDType,
    });
}

export class Chat {
    private messages: Message[] = [];
    private generator: TextGenerationPipeline;
    private readonly systemMessage: string;

    constructor(
        generator: TextGenerationPipeline,
        systemMessage: string = "You are an EdgeAI assistant for developers. Focus on local inference, WebGPU/WASM, latency, privacy, and practical coding tasks. Reply in simple English. Keep output short and structured for a small model. If a table is requested, output strict markdown table syntax with: 1 header row, 1 separator row using --- , and up to 4 data rows.",
    ) {
        this.generator = generator;
        this.systemMessage = systemMessage;
        this.messages.push({ role: "system", content: systemMessage });
    }

    async sendMessage(message: string, maxNewTokens: number = 512) {
        console.debug("Sending message:", message);
        console.debug("Current messages:", this.messages);

        this.messages.push({ role: "user", content: message });
        const generationOptions = {
            max_new_tokens: maxNewTokens,
            do_sample: true,
            temperature: 0.15,
            top_p: 0.85,
            repetition_penalty: 1.15,
            no_repeat_ngram_size: 3,
        };

        const chatMessages = this.buildApiMessages();
        console.debug("Using API chat messages:", chatMessages);

        const output = (await this.generator(
            chatMessages as any,
            generationOptions,
        )) as Array<{ generated_text?: unknown }>;

        console.debug("Generator output:", output);

        if (!output || !output[0] || !output[0].generated_text) {
            throw new Error("No output from generator");
        }

        const rawText = this.extractRawText(output[0].generated_text);
        const answer = this.extractAssistantAnswer(rawText);

        if (!answer) {
            throw new Error("Empty output from generator");
        }

        this.messages.push({
            role: "assistant",
            content: answer,
        });
        console.debug("Assistant answer:", answer);
        console.debug("Updated messages:", this.messages);

        return answer;
    }

    public getChatMessages() {
        return this.messages.filter((msg) => msg.role !== "system");
    }

    private buildApiMessages(): Message[] {
        const recentTurns = this.messages
            .filter((msg) => msg.role !== "system")
            .slice(-6);

        return [
            { role: "system", content: this.systemMessage },
            ...recentTurns,
        ];
    }

    private extractAssistantAnswer(rawText: string): string {
        console.debug("Extracting assistant answer from raw text:", rawText);

        const normalized = rawText
            .replace(
                /^\s*(Assistant|ASSISTENTE|USUARIO|User|System)\s*:\s*/i,
                "",
            )
            .trim();

        const clippedAtNextRole = normalized
            .split(
                /\n(?:\d+\.\s*(?:USUARIO|ASSISTENTE):|\[(?:INSTRUCOES|CONTEXTO|RESPOSTA)\]|(?:User|System|Assistant):)/,
            )[0]
            .trim();

        if (clippedAtNextRole.length > 0) {
            return clippedAtNextRole;
        }

        if (normalized.length > 0) {
            return normalized;
        }

        return rawText.trim();
    }

    private extractRawText(generatedText: unknown): string {
        if (typeof generatedText === "string") {
            return generatedText;
        }

        if (Array.isArray(generatedText)) {
            const last = generatedText.at(-1);

            if (typeof last === "string") {
                return last;
            }

            if (
                typeof last === "object" &&
                last !== null &&
                "content" in last &&
                typeof (last as { content?: unknown }).content === "string"
            ) {
                return (last as { content: string }).content;
            }
        }

        if (
            typeof generatedText === "object" &&
            generatedText !== null &&
            "content" in generatedText &&
            typeof (generatedText as { content?: unknown }).content === "string"
        ) {
            return (generatedText as { content: string }).content;
        }

        return String(generatedText ?? "");
    }
}
