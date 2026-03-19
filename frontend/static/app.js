function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMoney(value) {
  const amount = Number(value ?? 0);
  if (!Number.isFinite(amount)) return "$0.00";
  return `${amount >= 0 ? "+" : "-"}$${Math.abs(amount).toFixed(2)}`;
}

function statusBadge(value) {
  const text = escapeHtml(value || "unknown");
  const lowered = String(value || "").toLowerCase();
  let klass = "neutral";
  if (lowered.includes("reject") || lowered.includes("error") || lowered === "offline") klass = "bad";
  else if (lowered.includes("running") || lowered.includes("testing")) klass = "";
  else if (lowered.includes("idle") || lowered.includes("promot") || lowered.includes("updated")) klass = "warn";
  return `<span class="badge ${klass}">${text}</span>`;
}

function renderCard(label, value, subvalue = "") {
  return `
    <article class="card">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${value}</div>
      ${subvalue ? `<div class="subvalue">${subvalue}</div>` : ""}
    </article>
  `;
}

function renderStrategyCard(title, strategy, fallbackCopy) {
  if (!strategy) {
    return renderCard(title, "Unavailable", fallbackCopy || "No data available.");
  }
  const results = strategy.results_summary || {};
  const details = [
    `Version: ${escapeHtml(strategy.version ?? "-")}`,
    `Status: ${escapeHtml(strategy.latest_result_status || strategy.promotion_status || "-")}`,
    `Trades: ${escapeHtml(results.num_trades ?? 0)}`,
    `P&L: ${escapeHtml(formatMoney(results.total_pnl ?? 0))}`,
    `Win Rate: ${escapeHtml(results.win_rate ?? 0)}%`
  ].join("<br>");
  return renderCard(title, escapeHtml(strategy.id || "Unknown"), details);
}
