export class Timer {
    private startTime: number = 0;
    private endTime?: number;

    private constructor(ts: number) {
        this.startTime = ts;
    }

    static get(): Timer {
        return new Timer(Date.now());
    }

    static wait(ms: number): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    static wrap<T>(fn: () => Promise<T>): () => Promise<[T, number]> {
        return async () => {
            const timer = Timer.get();
            const result = await fn();
            const time = timer.end();
            return [result, time];
        };
    }

    end(): number {
        if (this.endTime) {
            return this.endTime - this.startTime;
        }

        this.endTime = Date.now();

        return this.end();
    }
}
