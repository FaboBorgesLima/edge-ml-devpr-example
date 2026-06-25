import { hasGPU } from "./has-gpu";

export async function hasFp16GpuSupport(): Promise<boolean> {
    if (!(await hasGPU())) return false;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    const device = await adapter.requestDevice();
    if (!device) return false;

    const fp16Supported = device.features.has("shader-f16");
    device.destroy();

    return fp16Supported;
}
