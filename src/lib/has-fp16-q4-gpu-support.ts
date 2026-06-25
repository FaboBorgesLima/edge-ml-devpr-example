import { hasFp16GpuSupport } from "./has-fp16-gpu-support";
import { hasGPU } from "./has-gpu";

export async function hasFp16Q4GPUSupport(): Promise<boolean> {
    if (!(await hasGPU())) return false;
    if (!(await hasFp16GpuSupport())) return false;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    const device = await adapter.requestDevice();
    if (!device) return false;

    const limits = device.limits;
    if (limits.maxComputeWorkgroupsPerDimension < 4) return false;

    return true;
}
