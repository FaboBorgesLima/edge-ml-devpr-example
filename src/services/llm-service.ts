import {
    pipeline,
    TextGenerationPipeline,
    type Message,
} from "@huggingface/transformers";

export async function getGenerator(useGpu: boolean) {
    return await pipeline(
        "text-generation",
        "onnx-community/gemma-3-270m-it-ONNX",
        {
            device: useGpu ? "webgpu" : "wasm",
            dtype: useGpu ? "q4f16" : "fp16",
        },
    );
}

export class Chat {
    private messages: Message[] = [];
    private generator: TextGenerationPipeline;
    private readonly systemMessage: string;

    constructor(
        generator: TextGenerationPipeline,
        systemMessage: string = "Você é um assistente de IA útil, prestativo e amigável.",
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
            do_sample: false,
            temperature: 0.2,
            top_p: 0.85,
            repetition_penalty: 1.2,
            no_repeat_ngram_size: 4,
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
