(function () {
    function copyText(text) {
        if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(text);
        }

        return new Promise(function (resolve, reject) {
            var textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.setAttribute("readonly", "");
            textArea.style.position = "fixed";
            textArea.style.top = "-9999px";
            document.body.appendChild(textArea);
            textArea.select();

            try {
                document.execCommand("copy");
                resolve();
            } catch (error) {
                reject(error);
            } finally {
                document.body.removeChild(textArea);
            }
        });
    }

    function addCopyButtons() {
        document.querySelectorAll("div.highlight").forEach(function (block) {
            if (block.querySelector(".eventlab-copy-button")) {
                return;
            }

            var pre = block.querySelector("pre");
            if (!pre) {
                return;
            }

            var button = document.createElement("button");
            button.type = "button";
            button.className = "eventlab-copy-button";
            button.textContent = "Copy";
            button.setAttribute("aria-label", "Copy code block");

            button.addEventListener("click", function () {
                var text = pre.innerText.replace(/\s+$/, "");
                copyText(text).then(function () {
                    button.textContent = "Copied";
                    button.classList.add("is-copied");
                    window.setTimeout(function () {
                        button.textContent = "Copy";
                        button.classList.remove("is-copied");
                    }, 1400);
                }).catch(function () {
                    button.textContent = "Failed";
                    window.setTimeout(function () {
                        button.textContent = "Copy";
                    }, 1400);
                });
            });

            block.appendChild(button);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", addCopyButtons);
    } else {
        addCopyButtons();
    }
}());
