import "./style.css";
import { render as renderEvaluation } from "./pages/evaluation";
import { render as renderLlm } from "./pages/llm";
import { render as renderVlm } from "./pages/vlm";
import { render } from "./pages/index";

document.addEventListener("DOMContentLoaded", main);

async function main() {
    const app = document.getElementById("app");

    const viteBase = import.meta.env.BASE_URL;

    if (window.location.pathname === `${viteBase}evaluation`) {
        return renderEvaluation(app!);
    }
    if (window.location.pathname === `${viteBase}vlm`) {
        return renderVlm(app!);
    }
    if (window.location.pathname === `${viteBase}llm`) {
        return renderLlm(app!);
    }
    return render(app!);
}
