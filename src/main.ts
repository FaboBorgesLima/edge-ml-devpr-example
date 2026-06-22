import "./style.css";
import { render } from "./pages/evaluation";

document.addEventListener("DOMContentLoaded", main);

async function main() {
    const app = document.getElementById("app");
    if (window.location.pathname === "/evaluation") {
        render(app!);
    }
}
