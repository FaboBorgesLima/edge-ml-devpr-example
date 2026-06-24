export async function hasGPU(): Promise<boolean> {
    // 1. Check if the WebGPU API exists in the browser
    if (!navigator.gpu) {
        return false;
    }

    try {
        // 2. Try to request the physical graphics adapter
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return false;
        }
        console.log("GPU Adapter found:", adapter.info);
        // 3. Try to request the logical device
        const device = await adapter.requestDevice();

        console.log("GPU Device created:", device);

        // Success! Clean up the device immediately
        device.destroy();
        return true;
    } catch (error) {
        console.error("Error while checking for GPU:", error);
        return false;
    }
}
