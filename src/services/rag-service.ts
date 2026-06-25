import { Chat, getGenerator, type AvailableModel } from "./llm-service";
import {
    rankBySemanticSimilarity,
    warmupSemanticModel,
    type SemanticHit,
} from "./semantic-service";

export interface RagAnswer {
    answer: string;
    retrieved: SemanticHit[];
}

export interface RagRuntime {
    ask(
        question: string,
        docs: string[],
        maxTokens?: number,
    ): Promise<RagAnswer>;
}

export async function createRagRuntime(
    useGpu: boolean,
    model: AvailableModel,
): Promise<RagRuntime> {
    await warmupSemanticModel(useGpu);
    const generator = await getGenerator(model);

    return {
        async ask(
            question: string,
            docs: string[],
            maxTokens: number = 220,
        ): Promise<RagAnswer> {
            const retrieved = await rankBySemanticSimilarity(
                question,
                docs,
                useGpu,
                3,
            );

            const context = retrieved
                .map((hit, idx) => `[${idx + 1}] ${hit.text}`)
                .join("\n");

            const ragPrompt = [
                "Task: answer using only the provided context.",
                "If the answer is not in context, say: Not found in provided context.",
                "Output format: 3 short bullets max.",
                "",
                `Question: ${question}`,
                "",
                "Context:",
                context || "(empty)",
            ].join("\n");

            const chat = new Chat(
                generator,
                "You are a strict RAG assistant. Use only the provided context. Do not invent facts.",
            );

            const answer = await chat.sendMessage(ragPrompt, maxTokens);
            return { answer, retrieved };
        },
    };
}
