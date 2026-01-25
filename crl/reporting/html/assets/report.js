(function () {
  const dataEl = document.getElementById("crl-report-data");
  const payload = dataEl ? JSON.parse(dataEl.textContent || "{}") : {};

  function download(filename, text, type) {
    const blob = new Blob([text], { type: type || "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  const jsonBtn = document.querySelector("[data-download-json]");
  if (jsonBtn) {
    jsonBtn.addEventListener("click", () => {
      download("report.json", JSON.stringify(payload, null, 2), "application/json");
    });
  }

  const csvBtn = document.querySelector("[data-download-csv]");
  if (csvBtn) {
    csvBtn.addEventListener("click", () => {
      const rows = payload.estimates || [];
      const headers = ["estimator", "value", "stderr", "lower_bound", "upper_bound"];
      const lines = [headers.join(",")];
      rows.forEach((row) => {
        const line = headers.map((key) => {
          const val = row[key];
          return typeof val === "string" ? `"${val.replace(/"/g, '""')}"` : (val ?? "");
        });
        lines.push(line.join(","));
      });
      download("summary.csv", lines.join("\n"), "text/csv");
    });
  }

  document.querySelectorAll("table[data-sortable]").forEach((table) => {
    const tbody = table.querySelector("tbody");
    const headers = table.querySelectorAll("thead th");
    headers.forEach((header, index) => {
      header.addEventListener("click", () => {
        const type = header.getAttribute("data-type") || "text";
        const rows = Array.from(tbody.querySelectorAll("tr"));
        const asc = !header.classList.contains("sorted-asc");
        headers.forEach((h) => h.classList.remove("sorted-asc", "sorted-desc"));
        header.classList.add(asc ? "sorted-asc" : "sorted-desc");
        rows.sort((a, b) => {
          const aText = a.children[index].textContent.trim();
          const bText = b.children[index].textContent.trim();
          if (type === "number") {
            const aNum = parseFloat(aText || "0");
            const bNum = parseFloat(bText || "0");
            return asc ? aNum - bNum : bNum - aNum;
          }
          return asc ? aText.localeCompare(bText) : bText.localeCompare(aText);
        });
        rows.forEach((row) => tbody.appendChild(row));
      });
    });
  });

  const modal = document.querySelector(".crl-modal");
  const modalImage = document.querySelector(".crl-modal-image");
  const modalTitle = document.querySelector(".crl-modal-title");
  const closeBtn = document.querySelector("[data-modal-close]");

  function closeModal() {
    if (!modal) return;
    modal.setAttribute("hidden", "");
    modal.hidden = true;
    modal.style.display = "none";
  }

  function openModal(src, title) {
    if (!modal || !modalImage || !modalTitle) return;
    modalImage.src = src || "";
    modalTitle.textContent = title || "Figure";
    modal.hidden = false;
    modal.removeAttribute("hidden");
    modal.style.display = "flex";
  }

  document.querySelectorAll("[data-figure-src]").forEach((btn) => {
    btn.addEventListener("click", () => {
      openModal(
        btn.getAttribute("data-figure-src"),
        btn.getAttribute("data-figure-title")
      );
    });
  });

  if (closeBtn) {
    closeBtn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      closeModal();
    });
  }

  if (modal) {
    modal.addEventListener("click", (event) => {
      if (event.target === modal) {
        closeModal();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModal();
    }
  });

  closeModal();
})();
