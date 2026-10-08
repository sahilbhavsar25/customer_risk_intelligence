const $ = (id) => document.getElementById(id);

let currentCustomer = null;

// --------------------------------------------------------------
// helpers
// --------------------------------------------------------------

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let body = null;
  try {
    body = await response.json();
  } catch {
    // non-JSON error
  }

  if (!response.ok) {
    const detail =
      body?.errors?.map((e) => e.message).join("; ") ||
      body?.detail ||
      `Request failed (${response.status})`;
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }

  return body;
}

function renderTable(table, rows, columns) {
  if (!rows.length) {
    table.innerHTML = `<tr><td class="muted">No records.</td></tr>`;
    return;
  }

  const head = columns.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows
    .map(
      (row) =>
        "<tr>" +
        columns
          .map((c) => `<td>${escapeHtml(c.format ? c.format(row[c.key]) : row[c.key] ?? "—")}</td>`)
          .join("") +
        "</tr>"
    )
    .join("");

  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
}

const money = (v) =>
  v == null ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// --------------------------------------------------------------
// health
// --------------------------------------------------------------

async function loadHealth() {
  try {
    const health = await api("/health");
    const pill = $("health");
    pill.textContent =
      health.status === "ok"
        ? `healthy · ${health.model} ${health.model_version}`
        : `degraded · ${Object.entries(health.components)
            .filter(([, v]) => v !== "ok" && v !== "configured")
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ")}`;
    pill.className = `pill ${health.status}`;
  } catch {
    $("health").textContent = "API unreachable";
  }
}

// --------------------------------------------------------------
// customer
// --------------------------------------------------------------

function renderProfile(customer) {
  const fields = [
    ["Customer ID", customer.customer_id],
    ["Name", customer.customer_name],
    ["Customer since", customer.customer_since],
    ["Segment", customer.customer_segment],
    ["Industry", customer.industry],
    ["Country", customer.country],
    ["Account status", customer.account_status],
  ];

  $("profile").innerHTML = fields
    .map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`)
    .join("");
}

async function loadRisk(customerId) {
  const target = $("risk");
  target.innerHTML = `<span class="muted">Loading…</span>`;

  try {
    const risk = await api("/predict-risk", {
      method: "POST",
      body: JSON.stringify({ customer_id: customerId }),
    });

    const factors = risk.risk_factors.length
      ? `<ul class="factors">${risk.risk_factors.map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul>`
      : `<p class="muted">No positive risk drivers identified.</p>`;

    target.innerHTML = `
      <div class="risk-level risk-${risk.risk_level}">${risk.risk_level}</div>
      <div>Probability: <strong>${(risk.risk_probability * 100).toFixed(1)}%</strong></div>
      <div class="muted">Snapshot ${risk.observation_date} · model ${escapeHtml(risk.model_name)} ${escapeHtml(risk.model_version)}</div>
      ${factors}`;
  } catch (error) {
    // 422 = valid customer without enough history; show the
    // message instead of an error state.
    target.innerHTML = `<p class="${error.status === 422 ? "muted" : "error"}">${escapeHtml(error.message)}</p>`;
  }
}

function renderHistory(data) {
  $("payment-summary").innerHTML =
    Object.entries(data.payment_summary)
      .map(([status, count]) => `<span class="pill">${status}: ${count}</span>`)
      .join("") || `<span class="muted">No transactions.</span>`;

  renderTable($("transactions"), data.recent_transactions, [
    { key: "transaction_date", label: "Date" },
    { key: "transaction_id", label: "Transaction" },
    { key: "transaction_amount", label: "Amount", format: money },
    { key: "payment_status", label: "Status" },
    { key: "due_date", label: "Due" },
    { key: "payment_date", label: "Paid" },
  ]);

  renderTable($("interactions"), data.recent_interactions, [
    { key: "interaction_date", label: "Date" },
    { key: "interaction_type", label: "Type" },
    { key: "channel", label: "Channel" },
    { key: "sentiment", label: "Sentiment" },
    { key: "resolution_status", label: "Resolution" },
  ]);

  $("documents").innerHTML = data.documents.length
    ? data.documents
        .map(
          (d) => `
        <div class="doc">
          <div class="doc-meta">${escapeHtml(d.document_id)} · ${escapeHtml(d.document_type)} · ${escapeHtml(d.document_date)} · ${escapeHtml(d.source)}</div>
          <div>${escapeHtml(d.content)}</div>
        </div>`
        )
        .join("")
    : `<p class="muted">No documents for this customer.</p>`;
}

async function searchCustomer(event) {
  event.preventDefault();

  const customerId = $("customer-id").value.trim().toUpperCase();
  $("search-error").hidden = true;
  $("answer").innerHTML = "";

  try {
    const data = await api(`/customers/${encodeURIComponent(customerId)}`);
    currentCustomer = data.customer.customer_id;

    renderProfile(data.customer);
    renderHistory(data);
    $("customer-view").hidden = false;

    loadRisk(currentCustomer);
  } catch (error) {
    $("customer-view").hidden = true;
    $("search-error").textContent = error.message;
    $("search-error").hidden = false;
  }
}

// --------------------------------------------------------------
// intelligence
// --------------------------------------------------------------

async function askQuestion(event) {
  event.preventDefault();
  if (!currentCustomer) return;

  const button = event.submitter;
  const question = $("question").value;
  button.disabled = true;
  $("answer").innerHTML = `<p class="muted">Thinking…</p>`;

  try {
    const result = await api("/customer-intelligence", {
      method: "POST",
      body: JSON.stringify({ customer_id: currentCustomer, question }),
    });

    const sources = result.sources.length
      ? result.sources
          .map((s) => `${escapeHtml(s.document_id)} (${escapeHtml(s.document_type)}, ${escapeHtml(s.document_date)})`)
          .join(", ")
      : "none (structured data / model output)";

    $("answer").innerHTML = `
      <div class="answer">${escapeHtml(result.answer)}</div>
      <div class="answer-meta">
        Grounded: <strong>${result.grounded ? "yes" : "no"}</strong> ·
        Sources: ${sources}
        ${result.error ? ` · <span class="error">${escapeHtml(result.error)}</span>` : ""}
      </div>`;
  } catch (error) {
    $("answer").innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  } finally {
    button.disabled = false;
  }
}

$("search-form").addEventListener("submit", searchCustomer);
$("ask-form").addEventListener("submit", askQuestion);
loadHealth();
