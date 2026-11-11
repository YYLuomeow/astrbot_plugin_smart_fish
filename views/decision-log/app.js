const bridge = window.AstrBotPluginView;
const listEl = document.getElementById("list");
const statusEl = document.getElementById("status");
const settingsEl = document.getElementById("settings");
const autoEl = document.getElementById("auto");
const modalEl = document.getElementById("modal");
const modalMetaEl = document.getElementById("modal-meta");
const modalBodyEl = document.getElementById("modal-body");

const REFRESH_MS = 5000;
let loading = false;
let latestRecords = [];

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatTime(seconds) {
  if (!seconds) return "-";
  const date = new Date(seconds * 1000);
  const pad = (num) => String(num).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(
    date.getDate(),
  )} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(
    date.getSeconds(),
  )}`;
}

function renderSettings(settings) {
  if (!settings) {
    settingsEl.textContent = "";
    return;
  }
  const parts = [];
  parts.push(`判断开关：${settings.enabled ? "已启用" : "已关闭"}`);
  parts.push(`模式：${settings.mode === "decision" ? "JEV 决策" : "LLM 判断"}`);
  parts.push(`阈值：${Number(settings.threshold).toFixed(2)}`);
  if (settings.mode === "decision") {
    parts.push(`平台：${settings.platform}`);
    parts.push(`模型：${settings.model || "-"}`);
  }
  if (settings.enabled) {
    parts.push(
      `上下文：${settings.context_count ? `${settings.context_count} 条` : "不携带"}`,
    );
    parts.push(`图转文：${settings.image_to_text ? "已开启" : "已关闭"}`);
  }
  settingsEl.textContent = parts.join(" · ");
}

function renderRecord(record, index) {
  const hasError = Boolean(record.error);
  const resultClass = hasError ? "error" : record.result ? "ok" : "no";
  const badgeClass = hasError
    ? "result-error"
    : record.result
      ? "result-ok"
      : "result-no";
  const badgeText = hasError ? "判断失败" : record.result ? "会回复" : "不回复";

  const meta = [];
  const modeText =
    record.mode === "decision"
      ? "JEV 决策"
      : record.mode === "image"
        ? "图片消息"
        : "LLM 判断";
  meta.push(
    `<span class="badge ${badgeClass}">${badgeText}</span>`,
    `<span class="badge">${modeText}</span>`,
  );
  if (record.reason) {
    meta.push(`<span class="badge">${escapeHtml(record.reason)}</span>`);
  }
  if (record.platform) {
    meta.push(`<span class="badge">${escapeHtml(record.platform)}</span>`);
  }
  if (record.model) {
    meta.push(`<span class="badge">${escapeHtml(record.model)}</span>`);
  }
  meta.push(`<time>${formatTime(record.time)}</time>`);
  if (typeof record.probability === "number") {
    meta.push(
      `<span class="metric">概率 ${record.probability.toFixed(3)} / 阈值 ${Number(
        record.threshold ?? 0,
      ).toFixed(2)}</span>`,
    );
  }
  meta.push(`<span class="metric">耗时 ${record.duration_ms ?? "-"} ms</span>`);

  let bar = "";
  if (typeof record.probability === "number") {
    const percent = Math.min(Math.max(record.probability, 0), 1) * 100;
    const mark = Math.min(Math.max(Number(record.threshold ?? 0), 0), 1) * 100;
    bar = `<div class="bar"><span style="width:${percent}%"></span><i style="left:calc(${mark}% - 1px)"></i></div>`;
  }

  const blocks = [];
  const imageText = record.request?.image_text || "";
  if (imageText) {
    blocks.push(`
      <div class="block">
        <h4>图片转述</h4>
        <p>${escapeHtml(imageText)}</p>
      </div>
    `);
  }
  if (record.reply) {
    blocks.push(`
      <div class="block reply">
        <h4>回复</h4>
        <p>${escapeHtml(record.reply)}</p>
      </div>
    `);
  }

  return `
    <article class="card ${resultClass}">
      <div class="row">${meta.join("")}</div>
      <div class="sender">
        <strong>${escapeHtml(record.sender || "未知发送者")}</strong>
        · ${escapeHtml(record.session || "")}
        <br />${escapeHtml(record.message || "")}
      </div>
      ${blocks.join("")}
      ${bar}
      ${hasError ? `<div class="error-text">${escapeHtml(record.error)}</div>` : ""}
      <button type="button" class="detail-btn" data-idx="${index}">
        请求参数与原始返回
      </button>
    </article>
  `;
}

function openDetail(index) {
  const record = latestRecords[index];
  if (!record) return;
  modalMetaEl.textContent = `${formatTime(record.time)} · ${
    record.sender || "未知发送者"
  } · ${record.session || ""}`;
  modalBodyEl.innerHTML = `
    <div>
      <h3>请求参数</h3>
      <pre>${escapeHtml(JSON.stringify(record.request ?? {}, null, 2))}</pre>
    </div>
    <div>
      <h3>原始返回</h3>
      <pre>${escapeHtml(record.response || "（无返回内容）")}</pre>
    </div>
  `;
  modalEl.classList.add("open");
}

function closeDetail() {
  modalEl.classList.remove("open");
}

async function load() {
  if (!bridge || loading) return;
  loading = true;
  try {
    const data = await bridge.apiGet("decision/records", { limit: 100 });
    const records = Array.isArray(data?.records) ? data.records : [];
    latestRecords = records;
    renderSettings(data?.settings);
    statusEl.textContent = records.length
      ? `共 ${data.total} 条记录，显示最近 ${records.length} 条`
      : "暂无决策记录，发送一条群消息后即可看到判断结果";
    listEl.innerHTML = records.length
      ? records.map(renderRecord).join("")
      : '<div class="empty">暂无决策记录</div>';
  } catch (error) {
    statusEl.textContent = `加载失败：${error.message}`;
  } finally {
    loading = false;
  }
}

async function clearRecords() {
  if (!bridge) return;
  try {
    await bridge.apiPost("decision/records/clear", {});
    await load();
  } catch (error) {
    statusEl.textContent = `清空失败：${error.message}`;
  }
}

document.getElementById("refresh").addEventListener("click", load);
document.getElementById("clear").addEventListener("click", clearRecords);
document.getElementById("modal-close").addEventListener("click", closeDetail);

// Records are re-rendered on every auto refresh, so details live in a modal
// that keeps its open state instead of an inline <details> element.
listEl.addEventListener("click", (event) => {
  const button = event.target.closest(".detail-btn");
  if (!button) return;
  openDetail(Number(button.dataset.idx));
});

modalEl.addEventListener("click", (event) => {
  if (event.target === modalEl) closeDetail();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDetail();
});

let timer = setInterval(load, REFRESH_MS);
autoEl.addEventListener("change", () => {
  clearInterval(timer);
  if (autoEl.checked) {
    timer = setInterval(load, REFRESH_MS);
  }
});

window.addEventListener("beforeunload", () => clearInterval(timer));

if (bridge) {
  load();
  Promise.resolve()
    .then(() => bridge.ready?.())
    .then(load)
    .catch(() => {});
} else {
  statusEl.textContent = "未找到 AstrBot bridge，请在插件详情页中打开该视图";
}
