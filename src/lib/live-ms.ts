export interface LiveMsController {
    stop(finalMs?: number): number;
}

/**
 * PT: Atualiza um elemento com o tempo decorrido em tempo real para dar feedback de progresso.
 * EN: Updates an element with elapsed time in real-time to provide progress feedback.
 */
export function startLiveMs(
    target: HTMLElement,
    intervalMs: number = 20,
): LiveMsController {
    const start = performance.now();

    const render = () => {
        const elapsed = Math.round(performance.now() - start);
        target.innerText = `Tempo: ${elapsed}ms`;
    };

    render();
    const timer = window.setInterval(render, intervalMs);

    return {
        stop(finalMs?: number) {
            window.clearInterval(timer);
            const elapsed =
                typeof finalMs === "number"
                    ? Math.round(finalMs)
                    : Math.round(performance.now() - start);
            target.innerText = `Tempo: ${elapsed}ms`;
            return elapsed;
        },
    };
}
