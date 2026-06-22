export function resizeTextArea(textArea: HTMLTextAreaElement) {
    textArea.style.height = "auto";
    textArea.style.height = `${Math.min(textArea.scrollHeight)}px`;
}
