const API_BASE_URL = document.querySelector('meta[name="verillm-api-base"]')?.content.trim() || window.location.origin;
const ROUTES = new Set([
  'overview', 'verify', 'claims', 'evidence', 'hallucination', 'reliability',
  'explainability', 'retrieval', 'nli', 'comparison', 'regression', 'benchmarks',
  'datasets', 'errors', 'adversarial', 'experiments', 'pipeline', 'api', 'health',
  'settings', 'docs'
]);
const TITLES = {
  overview: 'Overview',
  verify: 'Verify a response',
  claims: 'Claim explorer',
  evidence: 'Evidence explorer',
  hallucination: 'Hallucination analysis',
  reliability: 'Reliability analytics',
  explainability: 'Explainability',
  retrieval: 'Retrieval evaluation',
  nli: 'NLI verification',
  comparison: 'Model comparison',
  regression: 'Regression monitoring',
  benchmarks: 'Benchmark explorer',
  datasets: 'Datasets',
  errors: 'Error analysis',
  adversarial: 'Adversarial tests',
  experiments: 'Experiments',
  pipeline: 'Pipeline architecture',
  api: 'API reference',
  health: 'System health',
  settings: 'Settings',
  docs: 'Documentation'
};
const VERDICTS = ['SUPPORTED', 'CONTRADICTED', 'UNSUPPORTED'];
const SEVERITIES = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
const TAXONOMIES = [
  'FABRICATION', 'CONTRADICTION', 'NUMERICAL_ERROR', 'TEMPORAL_ERROR',
  'ENTITY_CONFUSION', 'SOURCE_MISMATCH', 'CONTEXT_MISINTERPRETATION',
  'OTHER', 'UNKNOWN'
];

const settingsDefaults = {
  theme: 'cream',
  density: 'comfortable',
  motion: 'normal',
  topK: 3,
  minRetrievalScore: 0.5,
  lowConfidenceReviewThreshold: 0.5
};

const state = {
  page: 'overview',
  root: null,
  result: null,
  response: 'The Eiffel Tower is in Paris. OpenAI was founded in 2015. The Great Wall of China is across northern China.',
  selectedClaimId: null,
  overview: null,
  settings: loadSettings()
};

function loadSettings() {
  try {
    return { ...settingsDefaults, ...JSON.parse(localStorage.getItem('verillm-settings') || '{}') };
  } catch {
    return { ...settingsDefaults };
  }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[character]);
}

function titleCase(value) {
  return String(value || '').replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === '') return 'DATA NOT AVAILABLE';
  if (typeof value !== 'number' || !Number.isFinite(value)) return escapeHtml(value);
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(digits);
}

function formatRate(value) {
  return value === null || value === undefined ? 'DATA NOT AVAILABLE' : `${(Number(value) * 100).toFixed(1)}%`;
}

function pageHeader(title, description, source = '') {
  return `<header class="page-heading">
    <div><p class="eyebrow">${escapeHtml(state.page.toUpperCase())}</p>
    <h1>${escapeHtml(title)}</h1>
    <p>${escapeHtml(description)}</p></div>
    ${source ? `<span class="source-chip">${escapeHtml(source)}</span>` : ''}
  </header>`;
}

function panel(title, content, source = '') {
  return `<section class="panel data-panel">
    <div class="section-heading"><h2>${escapeHtml(title)}</h2>${source ? `<span class="source-label">${escapeHtml(source)}</span>` : ''}</div>
    ${content}
  </section>`;
}

function metricCard(label, value, note = '') {
  return `<article class="metric-card"><span>${escapeHtml(label)}</span><strong>${value}</strong>${note ? `<small>${escapeHtml(note)}</small>` : ''}</article>`;
}

function emptyState(message, detail = '') {
  return `<div class="empty-state"><strong>DATA NOT AVAILABLE</strong><p>${escapeHtml(message)}</p>${detail ? `<small>${escapeHtml(detail)}</small>` : ''}</div>`;
}

function errorState(error) {
  return `<div class="error-state"><strong>Unable to load this page</strong><p>${escapeHtml(error.message || error)}</p><button class="ghost-button" data-action="retry">Retry</button></div>`;
}

function loadingState(message = 'Loading page data…') {
  return `<div class="loading-state" role="status"><span class="loading-dot"></span>${escapeHtml(message)}</div>`;
}

function dataTable(headers, rows, emptyMessage = 'No records are available.') {
  if (!rows.length) return emptyState(emptyMessage);
  return `<div class="table-wrap"><table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join('')}</tr></thead>
    <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function verdictBadge(verdict) {
  const value = String(verdict || 'UNKNOWN').toUpperCase();
  return `<span class="verdict-badge verdict-${escapeHtml(value.toLowerCase())}">${escapeHtml(value)}</span>`;
}

function severityBadge(severity) {
  const value = String(severity || 'UNKNOWN').toUpperCase();
  return `<span class="severity-badge severity-${escapeHtml(value.toLowerCase())}">${escapeHtml(value)}</span>`;
}

function statusBadge(status) {
  const value = String(status || 'UNKNOWN').toUpperCase();
  return `<span class="status-badge status-${escapeHtml(value.toLowerCase())}">${escapeHtml(value)}</span>`;
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) }
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the HTTP status when the server did not return JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

async function ensureLiveResult() {
  if (state.result) return state.result;
  const result = await api('/api/overview');
  state.result = result;
  state.response = result.response || state.response;
  return result;
}

function pageFromHash() {
  const route = window.location.hash.replace(/^#\/?/, '').split('?')[0];
  return ROUTES.has(route) ? route : 'overview';
}

function setActiveNavigation(page) {
  document.querySelectorAll('.nav-item[data-page]').forEach((link) => {
    const active = link.dataset.page === page;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}

function navigate(page) {
  if (!ROUTES.has(page)) page = 'overview';
  if (window.location.hash !== `#/${page}`) window.location.hash = `/${page}`;
  else renderRoute();
}

async function renderRoute() {
  state.page = pageFromHash();
  setActiveNavigation(state.page);
  window.scrollTo({ top: 0, behavior: state.settings.motion === 'reduced' ? 'auto' : 'smooth' });
  state.root.innerHTML = `${pageHeader(TITLES[state.page], 'Measure, explain, and improve LLM reliability.')}<div class="page-body">${loadingState()}</div>`;
  document.title = `${TITLES[state.page]} | VeriLLM`;
  try {
    const html = await renderPage(state.page);
    if (state.page === pageFromHash()) {
      state.root.innerHTML = html;
      bindPageEvents(state.page);
    }
  } catch (error) {
    state.root.innerHTML = `${pageHeader(TITLES[state.page], 'Measure, explain, and improve LLM reliability.')}<div class="page-body">${errorState(error)}</div>`;
  }
}

async function renderPage(page) {
  switch (page) {
    case 'overview': return renderOverview();
    case 'verify': return renderVerify();
    case 'claims': return renderClaims();
    case 'evidence': return renderEvidence();
    case 'hallucination': return renderHallucination();
    case 'reliability': return renderReliability();
    case 'explainability': return renderExplainability();
    case 'retrieval': return renderRetrieval();
    case 'nli': return renderNli();
    case 'comparison': return renderComparison();
    case 'regression': return renderRegression();
    case 'benchmarks': return renderBenchmarks();
    case 'datasets': return renderDatasets();
    case 'errors': return renderErrors();
    case 'adversarial': return renderAdversarial();
    case 'experiments': return renderExperiments();
    case 'pipeline': return renderPipeline();
    case 'api': return renderApi();
    case 'health': return renderHealth();
    case 'settings': return renderSettings();
    case 'docs': return renderDocs();
    default: return emptyState('No renderer is available for this route.');
  }
}

async function renderOverview() {
  const persisted = await api('/api/page/overview');
  let live = null;
  let liveError = null;
  try {
    live = await ensureLiveResult();
  } catch (error) {
    liveError = error;
  }
  state.overview = persisted;
  const summary = live?.summary || {};
  const claims = live?.claims || [];
  const reliability = deriveLiveReliability(claims);
  const liveCount = (field) => liveError ? 'DATA NOT AVAILABLE' : formatNumber(summary[field] || 0);
  const metrics = [
    metricCard('Persisted evaluations', formatNumber(persisted.persisted_evaluations)),
    metricCard('Claims analyzed (current response)', liveCount('total_claims')),
    metricCard('Supported claims', liveCount('supported')),
    metricCard('Contradicted claims', liveCount('contradicted')),
    metricCard('Unsupported claims', liveCount('unsupported')),
    metricCard('High severity issues', liveCount('high_severity_count')),
    metricCard('Human review required', liveCount('human_review_count')),
    metricCard('Evidence retrieved (current response)', liveError ? 'DATA NOT AVAILABLE' : formatNumber(claims.reduce((count, claim) => count + (claim.evidence || []).length, 0)))
  ].join('');
  const recent = (persisted.recent_evaluations || []).map((item) => [
    escapeHtml(item.name), escapeHtml(item.dataset || 'Not recorded'),
    formatNumber(item.metrics.accuracy), formatNumber(item.metrics.macro_f1),
    escapeHtml(item.modified_at || 'Not recorded')
  ]);
  const datasetRows = (persisted.datasets || []).map((item) => [
    escapeHtml(item.dataset), escapeHtml(item.availability),
    formatNumber(item.records), escapeHtml(Object.keys(item.labels || {}).join(', ') || 'Not recorded')
  ]);
  return `${pageHeader('Reliability overview', 'Live response-level pipeline output is shown separately from persisted benchmark evaluations.', 'LIVE PIPELINE + PERSISTED ARTIFACTS')}
    <div class="metrics-grid">${metrics}</div>
    ${panel('Reliability overview — current live response', `<div class="metrics-grid compact-grid">
      ${metricCard('Support rate', formatRate(reliability.support_rate))}
      ${metricCard('Contradiction rate', formatRate(reliability.contradiction_rate))}
      ${metricCard('Unsupported rate', formatRate(reliability.unsupported_rate))}
      ${metricCard('Evidence coverage', formatRate(reliability.evidence_coverage))}
      ${metricCard('Mean verification confidence', reliability.mean_verification_confidence === null ? 'DATA NOT AVAILABLE' : formatRate(reliability.mean_verification_confidence))}
      ${metricCard('High severity rate', formatRate(reliability.high_severity_error_rate))}
      ${metricCard('Human review rate', formatRate(reliability.human_review_rate))}
    </div>${liveError ? `<div class="error-state"><strong>LIVE PIPELINE UNAVAILABLE</strong><p>${escapeHtml(liveError.message)}</p></div>` : ''}
      <p class="subtle-note">Derived from the current response’s actual live pipeline claims; not benchmark ground truth.</p>`, 'LIVE PIPELINE')}
    ${panel('Persisted evaluation artifacts', dataTable(['Evaluation', 'Dataset', 'Accuracy', 'Macro F1', 'Artifact modified'], recent, 'No classifier metric artifacts are persisted.'), 'results/metrics/')}
    ${panel('Processed dataset availability', dataTable(['Dataset', 'Status', 'Records', 'Labels'], datasetRows, 'Processed dataset files were not found.'), 'datasets/processed/')}`;
}

function deriveLiveReliability(claims) {
  const total = claims.length;
  const rate = (test) => total ? claims.filter(test).length / total : null;
  const confidence = claims.filter((claim) => claim.confidence !== null && claim.confidence !== undefined).map((claim) => Number(claim.confidence));
  return {
    support_rate: rate((claim) => claim.verdict === 'SUPPORTED'),
    contradiction_rate: rate((claim) => claim.verdict === 'CONTRADICTED'),
    unsupported_rate: rate((claim) => claim.verdict === 'UNSUPPORTED'),
    evidence_coverage: rate((claim) => (claim.evidence || []).length > 0),
    mean_verification_confidence: confidence.length ? confidence.reduce((sum, value) => sum + value, 0) / confidence.length : null,
    high_severity_error_rate: rate((claim) => claim.severity === 'HIGH'),
    human_review_rate: rate((claim) => claim.needs_human_review === true)
  };
}

function resultClaimCards(claims, { expandable = true } = {}) {
  if (!claims.length) return emptyState('The current response contains no extracted claims.');
  return `<div class="claim-list">${claims.map((claim) => {
    const nliConfidence = claim.nli_confidence === null || claim.nli_confidence === undefined
      ? 'Not evaluated'
      : `${(Number(claim.nli_confidence) * 100).toFixed(1)}%`;
    const pipelineConfidence = claim.confidence === null || claim.confidence === undefined
      ? 'Not evaluated'
      : `${(Number(claim.confidence) * 100).toFixed(1)}%`;
    const details = `<div class="claim-meta">
      <span>NLI confidence: ${nliConfidence}</span><span>Weighted pipeline confidence: ${pipelineConfidence}</span>
      <span>Retrieval similarity: ${formatRate(Number(claim.retrieval_score || 0))}</span>
      <span>NLI run: ${claim.nli_evaluated ? escapeHtml(claim.nli_label || 'completed') : 'Not run — no passage met the relevance threshold'}</span>
      <span>Type: ${escapeHtml(titleCase(claim.hallucination_type))}</span>
    </div>
    <p class="claim-text">${escapeHtml(claim.explanation || '')}</p>
    <p><strong>Severity reasons</strong></p><ul>${(claim.severity_reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join('')}</ul>
    <p><strong>Review reasons</strong></p><ul>${(claim.review_reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join('') || '<li>No review reason recorded.</li>'}</ul>
    ${renderEvidenceList(claim)}`;
    return `<details class="claim-item" ${expandable ? '' : 'open'}>
      <summary class="claim-header"><span class="claim-title">${escapeHtml(claim.claim_id)} · ${escapeHtml(claim.claim)}</span>
      ${verdictBadge(claim.verdict)} ${severityBadge(claim.severity)}</summary>
      ${details}
    </details>`;
  }).join('')}</div>`;
}

function renderEvidenceList(claim) {
  const evidence = claim.evidence || [];
  if (!evidence.length) return emptyState('No passages were retrieved for this claim.');
  return `<div class="evidence-list">${evidence.map((item, index) => {
    const score = Number(item.score || 0);
    const status = score >= Number(claim.min_retrieval_score ?? state.settings.minRetrievalScore)
      ? 'RELEVANT'
      : score > 0 ? 'WEAK' : 'INSUFFICIENT';
    return `<article class="evidence-item"><div class="claim-header">
      <strong>Evidence #${index + 1} · ${escapeHtml(item.evidence_id || 'ID unavailable')}</strong>
      <span>${formatRate(score)}</span>${statusBadge(status)}
      </div><p>${escapeHtml(item.evidence || '')}</p></article>`;
  }).join('')}</div>`;
}

async function renderVerify() {
  return `${pageHeader('Analyze an LLM response', 'Run the existing claim extraction, evidence retrieval, relevance gate, NLI verification, and scoring pipeline.', 'LIVE PIPELINE')}
    ${panel('Model response', `<form id="verify-form" class="verify-form">
      <label for="response-input">Response text</label>
      <textarea id="response-input" name="response" rows="7" required>${escapeHtml(state.response)}</textarea>
      <div class="form-grid">
        <label>Model name <input name="model_name" placeholder="Optional model identifier"></label>
        <label>Application <input name="application" placeholder="Optional application"></label>
        <label>Version <input name="version" placeholder="Optional version"></label>
      </div>
      <details class="settings-disclosure"><summary>Pipeline controls (apply to this request)</summary>
        <div class="form-grid">
          <label>Top-K evidence <input name="top_k" type="number" min="1" max="10" value="${Number(state.settings.topK)}"></label>
          <label>Minimum retrieval similarity <input name="min_retrieval_score" type="number" min="0" max="1" step="0.01" value="${Number(state.settings.minRetrievalScore)}"></label>
          <label>Low-confidence review threshold <input name="low_confidence_review_threshold" type="number" min="0" max="1" step="0.01" value="${Number(state.settings.lowConfidenceReviewThreshold)}"></label>
        </div>
      </details>
      <div class="form-actions"><button class="primary-button" type="submit">Analyze response</button>
      <span id="verify-state" class="source-label" aria-live="polite"></span></div>
    </form>`, 'POST /api/verify')}
    <div id="verify-output">${state.result ? renderAnalysisResult(state.result) : emptyState('Submit a response to run the live pipeline.', 'Model inference will not be replaced with a frontend mock.')}</div>`;
}

function renderAnalysisResult(result) {
  const summary = result.summary || {};
  const claims = result.claims || [];
  return `<div class="metrics-grid compact-grid">
    ${metricCard('Claims', formatNumber(summary.total_claims || 0))}
    ${metricCard('Supported', formatNumber(summary.supported || 0))}
    ${metricCard('Contradicted', formatNumber(summary.contradicted || 0))}
    ${metricCard('Unsupported', formatNumber(summary.unsupported || 0))}
    ${metricCard('Human review', formatNumber(summary.human_review_count || 0))}
    ${metricCard('Mode', escapeHtml(result.mode || 'Not recorded'))}
  </div><p class="subtle-note">A claim marked UNSUPPORTED means retrieved evidence did not support verification; it is not automatically a hallucination.</p>
  ${resultClaimCards(claims)}`;
}

async function renderClaims() {
  const result = await ensureLiveResult();
  const claims = result.claims || [];
  const selected = claims.find((claim) => claim.claim_id === state.selectedClaimId);
  if (selected) {
    return `${pageHeader('Claim detail', 'Inspect fields emitted by the current live pipeline.', 'LIVE PIPELINE')}
      <button class="ghost-button" data-action="clear-claim">← Back to all claims</button>
      ${resultClaimCards([selected], { expandable: false })}`;
  }
  return `${pageHeader('Claim explorer', 'Search and filter claim-level outputs from the most recent live analysis.', 'LIVE PIPELINE')}
    ${panel('Filters', `<div class="filter-bar">
      <label>Search claims <input id="claim-search" type="search" placeholder="Search claim text or ID"></label>
      <label>Verdict <select id="claim-verdict"><option value="">All verdicts</option>${VERDICTS.map((item) => `<option>${item}</option>`).join('')}</select></label>
      <label>Severity <select id="claim-severity"><option value="">All severities</option>${SEVERITIES.map((item) => `<option>${item}</option>`).join('')}</select></label>
      <label>Hallucination type <select id="claim-type"><option value="">All types</option>${TAXONOMIES.map((item) => `<option>${item}</option>`).join('')}</select></label>
      </div>`)}
    <div id="claims-table">${claimsTable(claims)}</div>`;
}

function claimsTable(claims) {
  const search = document.getElementById('claim-search')?.value.trim().toLowerCase() || '';
  const verdict = document.getElementById('claim-verdict')?.value || '';
  const severity = document.getElementById('claim-severity')?.value || '';
  const type = document.getElementById('claim-type')?.value || '';
  const filtered = claims.filter((claim) =>
    `${claim.claim_id} ${claim.claim}`.toLowerCase().includes(search) &&
    (!verdict || claim.verdict === verdict) &&
    (!severity || claim.severity === severity) &&
    (!type || claim.hallucination_type === type)
  );
  return dataTable(
    ['Claim ID', 'Claim', 'Verdict', 'Confidence', 'Retrieval', 'Severity', 'Hallucination type', 'Human review', 'Details'],
    filtered.map((claim) => [
      escapeHtml(claim.claim_id), escapeHtml(claim.claim), verdictBadge(claim.verdict),
      claim.confidence === null ? 'Not evaluated' : formatRate(claim.confidence),
      formatRate(Number(claim.retrieval_score || 0)), severityBadge(claim.severity),
      escapeHtml(titleCase(claim.hallucination_type)), claim.needs_human_review ? 'Required' : 'No',
      `<button class="table-action" data-claim-id="${escapeHtml(claim.claim_id)}">Open</button>`
    ]),
    'No live pipeline claims match the selected filters.'
  );
}

async function renderEvidence() {
  const result = await ensureLiveResult();
  const claims = result.claims || [];
  return `${pageHeader('Retrieved evidence explorer', 'Inspect ranked TF-IDF passages and the project relevance gate. Retrieved evidence is not proof of truth.', 'LIVE PIPELINE')}
    ${claims.length ? claims.map((claim) => `<section class="panel data-panel">
      <div class="section-heading"><h2>${escapeHtml(claim.claim_id)} · ${escapeHtml(claim.claim)}</h2>${verdictBadge(claim.verdict)}</div>
      <p class="subtle-note">Evidence retrieval uses TF-IDF similarity. Gate: ${formatRate(Number(claim.min_retrieval_score ?? state.settings.minRetrievalScore))}; verification: ${claim.nli_evaluated ? 'NLI evaluated' : 'NLI not run because no passage met the gate'}.</p>
      ${renderEvidenceList(claim)}</section>`).join('') : emptyState('No pipeline claims are available. Run a response analysis first.')}`;
}

async function renderHallucination() {
  const [data, result] = await Promise.all([api('/api/page/hallucination'), ensureLiveResult()]);
  const h = data.halueval || {};
  const hMetrics = h.metrics;
  const hReport = h.reliability_report?.reliability;
  const rag = data.ragtruth?.summary;
  const liveClaims = result.claims || [];
  const taxonomyCounts = countBy(liveClaims, 'hallucination_type');
  const severityCounts = countBy(liveClaims, 'severity');
  return `${pageHeader('Hallucination analysis', 'Keep response-level benchmark labels separate from live claim-level pipeline taxonomy.', 'BENCHMARK + LIVE PIPELINE')}
    ${panel('HaluEval · response-level classifier', hMetrics ? `<div class="metrics-grid compact-grid">
      ${metricCard('Task', 'Response hallucination detection')}
      ${metricCard('Holdout predictions', formatNumber((hMetrics.per_class_metrics?.CORRECT?.support || 0) + (hMetrics.per_class_metrics?.HALLUCINATED?.support || 0)))}
      ${metricCard('Accuracy', formatRate(hMetrics.accuracy))}
      ${metricCard('Macro F1', formatRate(hMetrics.macro_f1))}
      ${metricCard('Ground-truth hallucination rate', formatRate(hReport?.ground_truth_hallucination_rate))}
      ${metricCard('Correct rate', hReport?.ground_truth_hallucination_rate == null ? 'DATA NOT AVAILABLE' : formatRate(1 - hReport.ground_truth_hallucination_rate))}
    </div><p class="subtle-note">Classifier prediction metrics and benchmark label distribution are different quantities. Sources: results/metrics/HaluEval_TFIDF_LogReg_metrics.json and stage6_halueval_reliability_report.json.</p>` : emptyState('No HaluEval classifier or reliability artifacts were found.'), 'PERSISTED HALUEVAL ARTIFACTS')}
    ${panel('RAGTruth · response labels and annotated spans', rag ? `<div class="metrics-grid compact-grid">
      ${metricCard('Responses', formatNumber(rag.response_count))}
      ${metricCard('Clean responses', formatNumber(rag.label_distribution?.CLEAN))}
      ${metricCard('Hallucinated responses', formatNumber(rag.label_distribution?.HALLUCINATED))}
      ${metricCard('Hallucinated response rate', formatRate(rag.hallucinated_response_rate))}
      ${metricCard('Annotated spans', formatNumber(rag.hallucination_span_count))}
      ${metricCard('Mean spans / response', formatNumber(rag.mean_spans_per_response))}
      ${metricCard('Mean span length', formatNumber(rag.mean_span_length))}
    </div><p class="subtle-note">${escapeHtml(rag.observation || '')} Source: results/metrics/ragtruth_summary.json.</p>` : emptyState('RAGTruth summary is not persisted.'), 'PERSISTED RAGTRUTH ARTIFACT')}
    ${panel('Live pipeline taxonomy', taxonomyCounts ? dataTable(['Type', 'Count'], Object.entries(taxonomyCounts).map(([key, value]) => [escapeHtml(titleCase(key)), formatNumber(value)])) : emptyState('No live pipeline claims are available.'), 'CURRENT RESPONSE')}
    ${panel('Live pipeline severity', Object.keys(severityCounts).length ? dataTable(['Severity', 'Count'], Object.entries(severityCounts).map(([key, value]) => [severityBadge(key), formatNumber(value)])) : emptyState('No live pipeline claims are available.'), 'RULE-BASED SCORING')}
    <p class="subtle-note">${escapeHtml(data.pipeline_taxonomy)}</p>`;
}

function countBy(items, field) {
  return items.reduce((result, item) => {
    const key = item[field] || 'UNKNOWN';
    result[key] = (result[key] || 0) + 1;
    return result;
  }, {});
}

async function renderReliability() {
  const data = await api('/api/page/reliability');
  const reports = (data.reports || []).filter(Boolean);
  if (!reports.length) return `${pageHeader('Reliability analytics', 'Persisted Stage 6 report metrics.', 'PERSISTED ARTIFACTS')}${emptyState('No reliability reports are available.', 'Expected results/metrics/stage6_*_reliability_report.json.')}`;
  return `${pageHeader('Reliability analytics', 'Report measured outcomes separately from rule-based risk and benchmark hallucination ground truth.', 'PERSISTED STAGE 6 REPORTS')}
    <div class="filter-bar"><label>Dataset <select id="reliability-dataset"><option value="">All available datasets</option>${reports.map((report) => `<option value="${escapeHtml(report.dataset)}">${escapeHtml(report.dataset)}</option>`).join('')}</select></label></div>
    <div id="reliability-reports">${reports.map(reliabilityReportCard).join('')}</div>
    ${panel('Experimental reliability score', `<p>${escapeHtml(data.metric_caveat)}</p><pre class="code-block">${escapeHtml(JSON.stringify(data.composite_config?.composite || null, null, 2))}</pre>`, 'configs/reliability_config.yaml')}`;
}

function reliabilityReportCard(report) {
  const r = report.reliability || {};
  return `<section class="panel data-panel reliability-report" data-dataset="${escapeHtml(report.dataset)}">
    <div class="section-heading"><h2>${escapeHtml(report.dataset)} · ${escapeHtml(report.task)}</h2><span>${formatNumber(report.sample_count)} records · ${escapeHtml(report.metadata?.split || 'split not recorded')}</span></div>
    <div class="metrics-grid compact-grid">
      ${metricCard('Support rate', formatRate(r.support_rate))}
      ${metricCard('Contradiction rate', formatRate(r.contradiction_rate))}
      ${metricCard('Unsupported rate', formatRate(r.unsupported_rate))}
      ${metricCard('Evidence coverage', formatRate(r.evidence_coverage))}
      ${metricCard('Mean verification confidence', r.mean_verification_confidence == null ? 'DATA NOT AVAILABLE' : formatRate(r.mean_verification_confidence))}
      ${metricCard('High severity rate', r.high_severity_error_rate == null ? 'DATA NOT AVAILABLE' : formatRate(r.high_severity_error_rate))}
      ${metricCard('Human review rate', r.human_review_rate == null ? 'DATA NOT AVAILABLE' : formatRate(r.human_review_rate))}
      ${metricCard('Experimental reliability score', r.composite_reliability_score == null ? 'DATA NOT AVAILABLE' : formatNumber(r.composite_reliability_score, 2))}
    </div><p class="subtle-note">Source: results/metrics/stage6_${escapeHtml(String(report.dataset).toLowerCase())}_reliability_report.json</p>
  </section>`;
}

async function renderExplainability() {
  const result = await ensureLiveResult();
  const claims = result.claims || [];
  const selected = claims.find((claim) => claim.claim_id === state.selectedClaimId) || claims[0];
  if (!selected) return `${pageHeader('Explanation flow', 'Deterministic explanation fields only; no hidden chain-of-thought is exposed.', 'LIVE PIPELINE')}${emptyState('No live claim explanation is available.', 'Submit a response from Verify to populate this page.')}`;
  return `${pageHeader('Explanation flow', 'Inspect the output path and project-produced explanation for one live claim.', 'LIVE PIPELINE')}
    <div class="filter-bar"><label>Selected claim <select id="explain-claim">${claims.map((claim) => `<option value="${escapeHtml(claim.claim_id)}" ${claim.claim_id === selected.claim_id ? 'selected' : ''}>${escapeHtml(claim.claim_id)} · ${escapeHtml(claim.claim)}</option>`).join('')}</select></label></div>
    <ol class="explanation-flow">
      ${[
        ['CLAIM', selected.claim],
        ['RETRIEVED EVIDENCE', (selected.evidence || []).map((entry) => entry.evidence).join('\n\n') || 'No evidence retrieved.'],
        ['SIMILARITY', (selected.evidence || []).map((entry, index) => `#${index + 1}: ${Number(entry.score).toFixed(4)}`).join(' · ') || 'Not available'],
        ['NLI VERDICT', selected.nli_evaluated ? `${selected.nli_label} → ${selected.verdict}` : 'NLI not run: no passage met the relevance threshold.'],
        ['CONFIDENCE', selected.nli_confidence == null ? 'NLI confidence not available because inference did not run.' : `NLI probability ${(selected.nli_confidence * 100).toFixed(1)}%; weighted pipeline score ${formatRate(selected.confidence)}`],
        ['HALLUCINATION ANALYSIS', titleCase(selected.hallucination_type)],
        ['SEVERITY', `${selected.severity}: ${(selected.severity_reasons || []).join(' ')}`],
        ['FINAL EXPLANATION', selected.explanation]
      ].map(([step, value]) => `<li class="flow-node"><span>${escapeHtml(step)}</span><p>${escapeHtml(value)}</p></li>`).join('')}
    </ol><p class="subtle-note">Explanation source: src/utils/explanations.py. These are deterministic/model-derived output fields, not private reasoning traces.</p>`;
}

async function renderRetrieval() {
  const data = await api('/api/page/retrieval');
  const retrieval = data.retrieval;
  const endToEnd = data.retrieval_nli;
  if (!retrieval) return `${pageHeader('Retrieval evaluation', 'Bounded evaluation using persisted Stage 7 artifacts.', 'SEED 42 · N=200')}
    ${emptyState('The Stage 7 retrieval comparison artifact is missing.', 'Expected results/metrics/stage7_retrieval_before_after.json.')}`;
  const datasets = Object.entries(retrieval).filter(([key, value]) => ['FEVER', 'AVeriTeC'].includes(key) && value?.original);
  return `${pageHeader('Retrieval evaluation', 'Exact whitespace-normalized gold-evidence Recall@K and retrieval-to-NLI results.', 'BOUNDED EVALUATION')}
    <p class="subtle-note">Configuration: seed ${data.configuration.seed}; ${data.configuration.sample_count_per_dataset} examples each from FEVER validation and AVeriTeC dev. AVeriTeC test data is excluded.</p>
    ${datasets.map(([name, block]) => {
      const versions = ['original', 'improved'].map((key) => block[key]);
      const rows = ['1', '3', '5', '10'].map((k) => [
        `Recall@${k}`, formatRate(versions[0]?.recall_at_k?.[k]), formatRate(versions[1]?.recall_at_k?.[k]),
        formatRate(block.absolute_delta?.recall_at_k?.[k])
      ]);
      rows.push(['Hit rate', formatRate(versions[0]?.evidence_hit_rate), formatRate(versions[1]?.evidence_hit_rate), formatRate(block.absolute_delta?.evidence_hit_rate)]);
      rows.push(['Failure rate', formatRate(versions[0]?.retrieval_failure_rate), formatRate(versions[1]?.retrieval_failure_rate), formatRate(block.absolute_delta?.retrieval_failure_rate)]);
      const nli = endToEnd?.[name];
      const nliRows = nli ? ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'].map((metric) => [
        titleCase(metric), formatRate(nli.original?.[metric]), formatRate(nli.improved?.[metric]), formatRate(nli.absolute_delta?.[metric])
      ]) : [];
      return `${panel(`${name} · Retrieval`, dataTable(['Metric', 'Original', 'Improved', 'Absolute delta'], rows), 'PERSISTED STAGE 7')}
        ${panel(`${name} · Retrieval followed by NLI`, nliRows.length ? dataTable(['Metric', 'Original', 'Improved', 'Absolute delta'], nliRows) : emptyState('No retrieval-to-NLI comparison is persisted.'))}`;
    }).join('')}
    <p class="subtle-note">${escapeHtml(retrieval.FEVer?.methodology || retrieval.FEVER?.methodology || '')}</p>`;
}

async function renderNli() {
  const data = await api('/api/page/nli');
  const items = Object.entries(data.metrics || {}).filter(([, metrics]) => metrics);
  if (!items.length) return `${pageHeader('NLI verification', 'Model, label mapping, and persisted NLI evaluation artifacts.', 'PERSISTED METRICS')}${emptyState('NLI evaluation artifacts are unavailable.')}`;
  return `${pageHeader('NLI verification', 'Evidence is the premise; extracted claim is the hypothesis. The verdict mapping is operational, not proof of truth.', 'MODEL')}
    ${panel('Model and operational mapping', `<h3>${escapeHtml(data.model || 'DATA NOT AVAILABLE')}</h3>
      ${dataTable(['NLI label', 'VeriLLM verdict'], Object.entries(data.mapping || {}).map(([label, verdict]) => [escapeHtml(label), verdictBadge(verdict)]))}
      <p class="subtle-note">${escapeHtml(data.confidence_distribution)}</p>`, 'configs/verification_config.yaml')}
    ${items.map(([dataset, metrics]) => `${panel(`${dataset} · persisted NLI metrics`, `<div class="metrics-grid compact-grid">
      ${metricCard('Accuracy', formatRate(metrics.accuracy))}
      ${metricCard('Macro precision', formatRate(metrics.macro_precision))}
      ${metricCard('Macro recall', formatRate(metrics.macro_recall))}
      ${metricCard('Macro F1', formatRate(metrics.macro_f1))}
      ${metricCard('Weighted F1', formatRate(metrics.weighted_f1))}
    </div><h3>Per-class F1</h3>${dataTable(['Class', 'Precision', 'Recall', 'F1', 'Support'], Object.entries(metrics.per_class_metrics || {}).map(([label, values]) => [escapeHtml(label), formatRate(values.precision), formatRate(values.recall), formatRate(values.f1), formatNumber(values.support)]))}
      <h3>Confusion matrix · rows actual, columns predicted</h3>${confusionMatrix(metrics.labels || [], metrics.confusion_matrix || [])}`, `results/metrics/NLI_${dataset}_InDomain_metrics.json`)}`).join('')}`;
}

function confusionMatrix(labels, matrix) {
  if (!labels.length || !matrix.length) return emptyState('Confusion matrix not persisted.');
  return dataTable(['Actual / Predicted', ...labels], matrix.map((row, index) => [escapeHtml(labels[index]), ...row.map(formatNumber)]));
}

async function renderComparison() {
  const data = await api('/api/page/comparison');
  const comparison = data.comparison;
  const entries = comparison?.entries || [];
  if (!entries.length) return `${pageHeader('Model comparison', 'Only compatible evaluations are compared; no model ranking is asserted.', 'FEVER VALIDATION')}${emptyState('No compatible FEVER comparison is persisted.')}`;
  const metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'];
  return `${pageHeader('FEVER model comparison', 'Side-by-side values and factual differences for compatible in-domain runs.', 'PERSISTED COMPARISON')}
    <p>${comparison.comparable ? 'Metadata marks these runs as comparable.' : escapeHtml(comparison.reason || 'Comparable metadata unavailable.')}</p>
    ${panel('Recorded model metrics', dataTable(['Model', ...metrics.map(titleCase)], entries.map((entry) => [escapeHtml(entry.model_name), ...metrics.map((metric) => formatRate(entry.metrics?.[metric]))])) , 'results/comparisons/fever_in_domain_models.json')}
    ${panel('Pairwise deltas (second entry minus first)', dataTable(['Metric', 'Absolute delta'], metrics.map((metric) => [titleCase(metric), formatRate(Number(entries[1]?.metrics?.[metric]) - Number(entries[0]?.metrics?.[metric]))])))}
    <p class="subtle-note">No winner or ranking is asserted. Available comparison files: ${escapeHtml((data.available_comparison_files || []).join(', '))}</p>`;
}

async function renderRegression() {
  const data = await api('/api/page/regression');
  const result = data.regression;
  if (!result) return `${pageHeader('Regression monitoring', 'Configured absolute-threshold comparison.', 'STAGE 6')}${emptyState('Historical regression comparison is unavailable.')}`;
  const rows = (result.comparisons || []).map((item) => [
    escapeHtml(titleCase(item.metric)),
    item.baseline_value == null ? 'Historical metric unavailable' : formatNumber(item.baseline_value),
    item.candidate_value == null ? 'Historical metric unavailable' : formatNumber(item.candidate_value),
    item.absolute_delta == null ? 'DATA NOT AVAILABLE' : formatNumber(item.absolute_delta),
    item.threshold == null ? 'DATA NOT AVAILABLE' : formatNumber(item.threshold),
    statusBadge(item.status),
    escapeHtml(item.reason || '')
  ]);
  return `${pageHeader('Regression monitoring', 'Preserves the persisted Stage 6 PASS/WARN/FAIL semantics, including historical data gaps.', 'STAGE 6 PERSISTED RESULT')}
    ${panel('Overall status', `<h2>${statusBadge(result.overall_regression_status)}</h2><p>${escapeHtml(result.methodology || '')}</p>`, 'NOT A SIGNIFICANCE TEST')}
    ${panel('Baseline vs candidate', dataTable(['Metric', 'Baseline', 'Candidate', 'Delta', 'Threshold', 'Status', 'Note'], rows), 'results/metrics/stage6_fever_logreg_to_svm_regression.json')}
    ${panel('Threshold configuration', `<pre class="code-block">${escapeHtml(JSON.stringify(data.configuration || null, null, 2))}</pre>`, 'configs/regression_config.yaml')}`;
}

async function renderBenchmarks() {
  const data = await api('/api/page/benchmarks');
  const claimDatasets = (data.datasets || []).filter((item) => item.task_type === 'CLAIM-LEVEL VERIFICATION');
  const responseDatasets = (data.datasets || []).filter((item) => item.task_type === 'RESPONSE-LEVEL HALLUCINATION DETECTION');
  const renderDatasetRows = (items) => dataTable(
    ['Dataset', 'Task', 'Task type', 'Processed samples', 'Labels', 'Splits', 'Role', 'Available metrics'],
    items.map((item) => [
      escapeHtml(item.dataset), escapeHtml(item.task), escapeHtml(item.task_type), formatNumber(item.records),
      escapeHtml(Object.entries(item.labels || {}).map(([key, count]) => `${key}: ${count}`).join(', ') || 'DATA NOT AVAILABLE'),
      escapeHtml(Object.entries(item.splits || {}).map(([key, count]) => `${key}: ${count}`).join(', ') || item.split_role),
      escapeHtml(item.project_usage), escapeHtml((item.metrics_available || []).join(', ') || 'None persisted')
    ])
  );
  return `${pageHeader('Benchmark explorer', 'Four task-aware datasets with distinct label semantics and evaluation roles.', 'NORMALIZED DATA + RESULTS')}
    ${panel('Claim-level verification', renderDatasetRows(claimDatasets), 'FEVER · AVERITEC')}
    ${panel('Response-level hallucination detection', renderDatasetRows(responseDatasets), 'HALUEVAL · RAGTRUTH')}
    <p class="subtle-note">Raw benchmark records are not modified. AVeriTeC evaluation uses train/dev only; its test split is excluded.</p>`;
}

async function renderDatasets() {
  const data = await api('/api/page/datasets');
  const datasets = data.datasets || [];
  return `${pageHeader('Dataset inventory', 'Raw benchmark source and processed canonical data are reported as separate read-only concepts.', 'READ ONLY')}
    <div class="dataset-grid">${datasets.map((item) => `<article class="panel dataset-card">
      <div class="section-heading"><h2>${escapeHtml(item.dataset)}</h2>${statusBadge(item.availability)}</div>
      <p>${escapeHtml(item.task)} · ${escapeHtml(item.task_type)}</p>
      <div class="dataset-columns">
        <div><h3>Raw dataset</h3><p>${escapeHtml(item.raw_dataset?.availability || 'DATA NOT AVAILABLE')}</p><p>${escapeHtml(item.raw_dataset?.note || '')}</p><p>UI editing: disabled</p></div>
        <div><h3>Processed dataset</h3><p>${escapeHtml(item.processed_dataset?.availability || 'DATA NOT AVAILABLE')}</p><p>${escapeHtml(item.processed_dataset?.path || 'No file path recorded')}</p><p>Records: ${formatNumber(item.records)}</p></div>
      </div>
      <h3>Labels</h3><ul>${Object.entries(item.labels || {}).map(([label, count]) => `<li>${escapeHtml(label)}: ${formatNumber(count)}</li>`).join('') || '<li>DATA NOT AVAILABLE</li>'}</ul>
      <p><strong>Splits:</strong> ${escapeHtml(Object.entries(item.splits || {}).map(([split, count]) => `${split} (${count})`).join(', ') || item.split_role)}</p>
      ${item.dataset === 'AVeriTeC' ? '<p class="safety-note">Test split isolation: processed normalization and project evaluation exclude AVeriTeC test.json.</p>' : ''}
      ${item.dataset === 'RAGTruth' ? '<p>Span analysis comes from the persisted RAGTruth summary; annotated spans remain separate from response labels.</p>' : ''}
    </article>`).join('')}</div>`;
}

async function renderErrors() {
  const data = await api('/api/page/errors');
  const files = data.files || [];
  const selected = data.selected_file;
  if (!selected) return `${pageHeader('Prediction error analysis', 'Rows are actual mismatches from persisted prediction JSONL artifacts.', 'PERSISTED PREDICTIONS')}${emptyState('No prediction artifacts are available.', 'Expected results/predictions/*_predictions.jsonl.')}`;
  const rows = (data.errors || []).map((item) => [
    escapeHtml(item.sample_id || '—'), escapeHtml(item.claim || item.evidence || '—'),
    escapeHtml(item.true_label || '—'), escapeHtml(item.predicted_label || '—'),
    item.confidence == null ? 'DATA NOT AVAILABLE' : formatRate(item.confidence),
    escapeHtml(item.evidence || 'DATA NOT AVAILABLE'),
    escapeHtml(item.source_dataset || 'Not recorded'),
    'DATA NOT AVAILABLE'
  ]);
  const matrix = data.metrics?.confusion_matrix || [];
  const labels = data.metrics?.labels || [];
  const classErrors = labels.map((label, index) => {
    const falseNegative = (matrix[index] || []).reduce((sum, value, column) => sum + (column === index ? 0 : value), 0);
    const falsePositive = matrix.reduce((sum, row, rowIndex) => sum + (row[index] || 0) - (rowIndex === index ? (row[index] || 0) : 0), 0);
    return [escapeHtml(label), formatNumber(falsePositive), formatNumber(falseNegative)];
  });
  return `${pageHeader('Prediction error analysis', 'Choose a persisted model output and inspect its false-positive/false-negative records.', 'NO SYNTHETIC ERRORS')}
    ${panel('Prediction artifact', `<div class="filter-bar"><label>Model/dataset artifact <select id="error-artifact">${files.map((name) => `<option ${name === selected ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('')}</select></label>
      <label>Expected label <select id="error-expected"><option value="">All</option>${VERDICTS.concat(['CONFLICTING_EVIDENCE', 'CORRECT', 'HALLUCINATED']).map((value) => `<option>${value}</option>`).join('')}</select></label>
      <label>Predicted label <select id="error-predicted"><option value="">All</option>${VERDICTS.concat(['CONFLICTING_EVIDENCE', 'CORRECT', 'HALLUCINATED']).map((value) => `<option>${value}</option>`).join('')}</select></label>
      <label>Minimum confidence <input id="error-confidence" type="number" min="0" max="1" step="0.01" value="0"></label>
      <label>Error type <select id="error-type"><option value="">All errors</option><option value="FP">False positive (per class)</option><option value="FN">False negative (per class)</option></select></label>
      <label>Error class <select id="error-class"><option value="">Select a class</option>${labels.map((label) => `<option>${escapeHtml(label)}</option>`).join('')}</select></label>
      </div><p>Selected source: results/predictions/${escapeHtml(selected)}. ${formatNumber(data.total_errors)} total mismatches; at most 500 rows are returned.</p>`, 'JSONL')}
    ${panel('Per-class error counts', dataTable(['Class', 'False positives', 'False negatives'], classErrors, 'This artifact does not contain a persisted confusion matrix.'), 'FROM CONFUSION MATRIX')}
    <div id="error-rows">${dataTable(['Sample ID', 'Input', 'Expected', 'Predicted', 'Confidence', 'Evidence/context', 'Dataset', 'Explanation'], rows, 'This persisted artifact contains no errors, or the mismatch set is empty.')}</div>`;
}

async function renderAdversarial() {
  const data = await api('/api/page/adversarial');
  return `${pageHeader('Adversarial and robustness results', 'Persisted synthetic tests, with failures shown explicitly.', 'SYNTHETIC FRAMEWORK EVALUATION')}
    <p class="warning-note">${escapeHtml(data.scope_note)}</p>
    ${(data.evaluations || []).map((evaluation) => {
      if (evaluation.availability !== 'AVAILABLE') return panel(evaluation.name, emptyState(evaluation.missing || 'Result artifact missing.'), 'DATA NOT AVAILABLE');
      const rows = evaluation.cases.map((test) => [
        escapeHtml(test.case), escapeHtml(test.category || 'Not recorded'), escapeHtml(test.input || 'Not recorded'),
        escapeHtml(test.expected || 'Not recorded'), escapeHtml(test.actual || 'Not recorded'),
        statusBadge(test.pass === true ? 'PASS' : test.pass === false ? 'FAIL' : 'DATA NOT AVAILABLE')
      ]);
      return panel(evaluation.name, `<div class="metrics-grid compact-grid">
        ${metricCard('Cases', formatNumber(evaluation.total_cases))}
        ${metricCard('Passed', formatNumber(evaluation.passed))}
        ${metricCard('Failed', formatNumber(evaluation.failed))}
        ${metricCard('Pass rate', formatRate(evaluation.pass_rate))}
      </div>${dataTable(['Case', 'Category', 'Input', 'Expected', 'Actual', 'Status'], rows)}`, 'PERSISTED RESULTS');
    }).join('')}`;
}

async function renderExperiments() {
  const data = await api('/api/page/experiments');
  const experiments = data.experiments || [];
  const rows = experiments.map((item) => [
    escapeHtml(item.name), escapeHtml(item.dataset || datasetFromName(item.name) || 'Not recorded'),
    escapeHtml(item.metrics.nli_model || 'Model not recorded'),
    formatNumber(item.sample_count), formatRate(item.metrics.accuracy), formatRate(item.metrics.macro_f1),
    escapeHtml(item.metrics.seed ?? 'Not recorded'), escapeHtml(item.modified_at || 'Date not recorded')
  ]);
  return `${pageHeader('Persisted experiment browser', 'Only metric JSON artifacts are shown. Missing metadata remains explicitly unavailable.', 'RESULTS/METRICS')}
    ${panel('Metric artifacts', dataTable(['Experiment', 'Dataset', 'Model', 'Samples', 'Accuracy', 'Macro F1', 'Seed', 'Artifact modified'], rows, 'No experiment metric artifacts were found.'), 'PERSISTED ARTIFACTS')}
    <p class="subtle-note">Configuration/date are not necessarily stored in metric files. The displayed filesystem modification time is artifact metadata, not a verified experiment run timestamp. Missing: ${(data.missing_metadata || []).map(escapeHtml).join(', ')}.</p>`;
}

function datasetFromName(name) {
  return ['FEVER', 'AVeriTeC', 'HaluEval', 'RAGTruth'].find((dataset) => String(name).toLowerCase().startsWith(dataset.toLowerCase())) || null;
}

async function renderPipeline() {
  const data = await api('/api/page/pipeline');
  const nodes = data.nodes || [];
  return `${pageHeader('VeriLLM analysis architecture', 'Explore the implemented data path; select a stage for its inputs, outputs, and source.', `PIPELINE ${escapeHtml(data.version?.pipeline_version || 'VERSION NOT RECORDED')}`)}
    <div class="pipeline-flow">${nodes.map((node, index) => `<button class="pipeline-node" data-node="${escapeHtml(node.id)}">
      <span>${escapeHtml(node.label)}</span>${index < nodes.length - 1 ? '<b aria-hidden="true">↓</b>' : ''}
    </button>`).join('')}</div>
    <div id="pipeline-detail" class="panel data-panel">${nodes[0] ? pipelineDetail(nodes[0]) : emptyState('Pipeline structure is unavailable.')}</div>`;
}

function pipelineDetail(node) {
  return `<p>${escapeHtml(node.purpose)}</p><dl class="detail-list">
    <dt>Implementation</dt><dd>${escapeHtml(node.implementation)}</dd>
    <dt>Source</dt><dd><code>${escapeHtml(node.source)}</code></dd>
    <dt>Inputs</dt><dd>${escapeHtml(node.inputs)}</dd>
    <dt>Outputs</dt><dd>${escapeHtml(node.outputs)}</dd>
  </dl>`;
}

async function renderApi() {
  const data = await api('/api/page/api');
  const rows = (data.routes || []).map((route) => [
    escapeHtml(route.methods.join(', ')), `<code>${escapeHtml(route.path)}</code>`, escapeHtml(route.description)
  ]);
  return `${pageHeader('FastAPI route reference', 'Generated from the running FastAPI application routes.', 'LIVE ROUTE TABLE')}
    ${panel('Registered API routes', dataTable(['Method', 'Path', 'Description'], rows, 'No API routes were registered.'), escapeHtml(data.base_url))}
    ${panel('POST /api/verify request schema', `<pre class="code-block">${escapeHtml(JSON.stringify(data.verify_request_schema || {}, null, 2))}</pre>
      <p>Response fields returned by the handler: ${escapeHtml((data.verify_response_fields || []).join(', '))}.</p>
      <p class="subtle-note">${escapeHtml(data.response_schema_note || '')}</p>`, 'GENERATED FROM FASTAPI REQUEST MODEL')}
    ${panel('Try POST /api/verify', `<form id="api-test-form" class="verify-form">
      <label for="api-test-input">Request JSON response text</label>
      <textarea id="api-test-input" rows="5">${escapeHtml(state.response)}</textarea>
      <button class="primary-button" type="submit">Send request</button>
      <pre id="api-test-output" class="code-block" aria-live="polite">Submit to display the actual API response.</pre>
    </form>`, 'POST /api/verify')}`;
}

async function renderHealth() {
  const data = await api('/api/health');
  const rows = Object.entries(data.checks || {}).map(([name, value]) => [
    escapeHtml(titleCase(name)),
    statusBadge(typeof value === 'string' ? value : value.status),
    escapeHtml(typeof value === 'string' ? value : value.detail || value.path || value.name || '')
  ]);
  return `${pageHeader('System health', 'Backend and local dependency checks reported by the API at request time.', data.status)}
    ${panel('Health checks', `<div class="section-heading"><h2>${statusBadge(data.status)}</h2><span>Last checked: ${escapeHtml(data.checked_at)}</span></div>
      ${dataTable(['Component', 'Status', 'Detail'], rows)}`, 'GET /api/health')}
    ${data.checks?.nli_model?.detail ? `<p class="warning-note">${escapeHtml(data.checks.nli_model.detail)}</p>` : ''}`;
}

function renderSettings() {
  const config = state.settings;
  return `${pageHeader('Product preferences', 'Appearance options persist locally. Evaluation controls are sent to /api/verify when supported.', 'LOCAL STORAGE + REQUEST SETTINGS')}
    ${panel('Appearance', `<div class="form-grid">
      <label>Theme <select name="theme"><option value="cream">Cream</option><option value="sky">Sky blue</option><option value="brown">Light brown</option></select></label>
      <label>Density <select name="density"><option value="comfortable">Comfortable</option><option value="compact">Compact</option></select></label>
      <label>Motion <select name="motion"><option value="normal">Normal</option><option value="reduced">Reduced</option></select></label>
      </div><button class="primary-button" data-action="save-settings">Save appearance</button>`, 'BROWSER LOCAL STORAGE')}
    ${panel('Evaluation request controls', `<div class="form-grid">
      <label>Top-K evidence <input id="setting-top-k" type="number" min="1" max="10" value="${Number(config.topK)}"></label>
      <label>Minimum retrieval similarity <input id="setting-retrieval" type="number" min="0" max="1" step="0.01" value="${Number(config.minRetrievalScore)}"></label>
      <label>Low-confidence review threshold <input id="setting-review" type="number" min="0" max="1" step="0.01" value="${Number(config.lowConfidenceReviewThreshold)}"></label>
      </div><p class="subtle-note">The retrieval threshold changes which passages reach NLI; the review threshold changes review flags. Settings do not alter persisted benchmark runs or the NLI model.</p>
      <button class="primary-button" data-action="save-evaluation-settings">Save evaluation controls</button>`, 'SENT WITH LIVE VERIFY REQUEST')}`;
}

async function renderDocs() {
  const data = await api('/api/page/docs');
  const documents = data.documents || [];
  return `${pageHeader('Project documentation', 'Architecture notes loaded directly from the repository documentation files.', 'docs/architecture/')}
    ${documents.map((document) => `<details class="panel doc-panel"><summary><strong>${escapeHtml(document.title)}</strong><span>${escapeHtml(document.path)}</span></summary>
      ${document.content ? `<div class="markdown-content">${renderMarkdown(document.content)}</div>` : emptyState(`Documentation file unavailable: ${document.path}`)}</details>`).join('')}`;
}

function renderMarkdown(markdown) {
  const escaped = escapeHtml(markdown);
  return escaped.split(/\r?\n/).map((line) => {
    if (/^### /.test(line)) return `<h4>${line.slice(4)}</h4>`;
    if (/^## /.test(line)) return `<h3>${line.slice(3)}</h3>`;
    if (/^# /.test(line)) return `<h2>${line.slice(2)}</h2>`;
    if (/^\|/.test(line)) return `<pre>${line}</pre>`;
    if (/^> /.test(line)) return `<blockquote>${line.slice(2)}</blockquote>`;
    if (/^- /.test(line)) return `<p>• ${line.slice(2)}</p>`;
    if (/^\d+\. /.test(line)) return `<p>${line}</p>`;
    if (line.startsWith('```')) return '';
    if (line.trim()) return `<p>${line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>')}</p>`;
    return '';
  }).join('');
}

function bindPageEvents(page) {
  state.root.querySelector('[data-action="retry"]')?.addEventListener('click', renderRoute);
  if (page === 'verify') {
    const form = state.root.querySelector('#verify-form');
    form?.addEventListener('submit', async (event) => {
      event.preventDefault();
      state.response = form.elements.response?.value ?? state.root.querySelector('#response-input').value;
      const status = state.root.querySelector('#verify-state');
      const output = state.root.querySelector('#verify-output');
      status.textContent = 'Running existing VeriLLM pipeline…';
      output.innerHTML = loadingState('Extracting claims, retrieving evidence, and invoking NLI where the relevance gate is met…');
      const payload = {
        response: state.root.querySelector('#response-input').value.trim(),
        top_k: Number(form.elements.top_k.value),
        min_retrieval_score: Number(form.elements.min_retrieval_score.value),
        low_confidence_review_threshold: Number(form.elements.low_confidence_review_threshold.value),
        model_name: form.elements.model_name.value.trim() || null,
        application: form.elements.application.value.trim() || null,
        version: form.elements.version.value.trim() || null
      };
      state.response = payload.response;
      try {
        state.result = await api('/api/verify', { method: 'POST', body: JSON.stringify(payload) });
        output.innerHTML = renderAnalysisResult(state.result);
        status.textContent = `${state.result.mode} · completed`;
      } catch (error) {
        output.innerHTML = errorState(error);
        status.textContent = 'Request failed';
      }
    });
  }
  if (page === 'claims') {
    const update = () => { state.root.querySelector('#claims-table').innerHTML = claimsTable(state.result?.claims || []); };
    ['#claim-search', '#claim-verdict', '#claim-severity', '#claim-type'].forEach((selector) => state.root.querySelector(selector)?.addEventListener('input', update));
    state.root.querySelector('#claims-table')?.addEventListener('click', (event) => {
      const button = event.target.closest('[data-claim-id]');
      if (!button) return;
      state.selectedClaimId = button.dataset.claimId;
      renderRoute();
    });
    state.root.querySelector('[data-action="clear-claim"]')?.addEventListener('click', () => {
      state.selectedClaimId = null;
      renderRoute();
    });
  }
  if (page === 'explainability') {
    state.root.querySelector('#explain-claim')?.addEventListener('change', (event) => {
      state.selectedClaimId = event.target.value;
      renderRoute();
    });
  }
  if (page === 'reliability') {
    state.root.querySelector('#reliability-dataset')?.addEventListener('change', (event) => {
      state.root.querySelectorAll('.reliability-report').forEach((report) => {
        report.hidden = Boolean(event.target.value && report.dataset.dataset !== event.target.value);
      });
    });
  }
  if (page === 'errors') {
    const reloadErrors = async () => {
      const file = state.root.querySelector('#error-artifact').value;
      const response = await api(`/api/page/errors?model=${encodeURIComponent(file)}`);
      const expected = state.root.querySelector('#error-expected').value;
      const predicted = state.root.querySelector('#error-predicted').value;
      const minimumConfidence = Number(state.root.querySelector('#error-confidence').value || 0);
      const type = state.root.querySelector('#error-type').value;
      const errorClass = state.root.querySelector('#error-class').value;
      const filtered = (response.errors || []).filter((item) =>
        (!expected || item.true_label === expected) &&
        (!predicted || item.predicted_label === predicted) &&
        (item.confidence == null || Number(item.confidence) >= minimumConfidence) &&
        (!type || (errorClass && (type === 'FP'
          ? item.predicted_label === errorClass && item.true_label !== errorClass
          : item.true_label === errorClass && item.predicted_label !== errorClass)))
      );
      state.root.querySelector('#error-rows').innerHTML = dataTable(
        ['Sample ID', 'Input', 'Expected', 'Predicted', 'Confidence', 'Evidence/context', 'Dataset', 'Explanation'],
        filtered.map((item) => [escapeHtml(item.sample_id || '—'), escapeHtml(item.claim || item.evidence || '—'), escapeHtml(item.true_label || '—'), escapeHtml(item.predicted_label || '—'), item.confidence == null ? 'DATA NOT AVAILABLE' : formatRate(item.confidence), escapeHtml(item.evidence || 'DATA NOT AVAILABLE'), escapeHtml(item.source_dataset || 'Not recorded'), 'DATA NOT AVAILABLE']),
        'No errors match these filters.'
      );
    };
    state.root.querySelector('#error-artifact')?.addEventListener('change', reloadErrors);
    state.root.querySelector('#error-expected')?.addEventListener('change', reloadErrors);
    state.root.querySelector('#error-predicted')?.addEventListener('change', reloadErrors);
    state.root.querySelector('#error-confidence')?.addEventListener('input', reloadErrors);
    state.root.querySelector('#error-type')?.addEventListener('change', reloadErrors);
    state.root.querySelector('#error-class')?.addEventListener('change', reloadErrors);
  }
  if (page === 'pipeline') {
    const nodes = state.root.querySelectorAll('[data-node]');
    const dataPromise = api('/api/page/pipeline');
    nodes.forEach((button) => button.addEventListener('click', async () => {
      const data = await dataPromise;
      const node = data.nodes.find((item) => item.id === button.dataset.node);
      if (node) state.root.querySelector('#pipeline-detail').innerHTML = pipelineDetail(node);
      nodes.forEach((item) => item.classList.toggle('selected', item === button));
    }));
  }
  if (page === 'api') {
    state.root.querySelector('#api-test-form')?.addEventListener('submit', async (event) => {
      event.preventDefault();
      const output = state.root.querySelector('#api-test-output');
      output.textContent = 'Sending request…';
      try {
        const result = await api('/api/verify', {
          method: 'POST',
          body: JSON.stringify({ response: state.root.querySelector('#api-test-input').value })
        });
        output.textContent = JSON.stringify(result, null, 2);
      } catch (error) {
        output.textContent = JSON.stringify({ error: error.message }, null, 2);
      }
    });
  }
  if (page === 'settings') {
    const controls = state.root;
    controls.querySelector('[name="theme"]').value = state.settings.theme;
    controls.querySelector('[name="density"]').value = state.settings.density;
    controls.querySelector('[name="motion"]').value = state.settings.motion;
    controls.querySelector('[data-action="save-settings"]')?.addEventListener('click', () => {
      state.settings.theme = controls.querySelector('[name="theme"]').value;
      state.settings.density = controls.querySelector('[name="density"]').value;
      state.settings.motion = controls.querySelector('[name="motion"]').value;
      saveSettings();
    });
    controls.querySelector('[data-action="save-evaluation-settings"]')?.addEventListener('click', () => {
      state.settings.topK = Math.max(1, Math.min(10, Number(controls.querySelector('#setting-top-k').value)));
      state.settings.minRetrievalScore = Math.max(0, Math.min(1, Number(controls.querySelector('#setting-retrieval').value)));
      state.settings.lowConfidenceReviewThreshold = Math.max(0, Math.min(1, Number(controls.querySelector('#setting-review').value)));
      saveSettings();
    });
  }
}

function saveSettings() {
  localStorage.setItem('verillm-settings', JSON.stringify(state.settings));
  applySettings();
  renderRoute();
}

function applySettings() {
  document.documentElement.dataset.theme = state.settings.theme;
  document.documentElement.dataset.density = state.settings.density;
  document.documentElement.dataset.motion = state.settings.motion;
}

function initialize() {
  const main = document.querySelector('.main-panel');
  const topbar = main.querySelector('.topbar');
  state.root = document.createElement('div');
  state.root.id = 'page-content';
  main.replaceChildren(topbar, state.root);
  document.querySelectorAll('.nav-item[data-page]').forEach((link) => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      navigate(link.dataset.page);
    });
    const topbarButtons = topbar.querySelectorAll('button');
    topbarButtons[0]?.addEventListener('click', () => {
      if (!state.result) return;
      const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'verillm-live-analysis.json';
      link.click();
      URL.revokeObjectURL(link.href);
    });
    topbarButtons[1]?.addEventListener('click', () => navigate('verify'));
  });
  window.addEventListener('hashchange', renderRoute);
  applySettings();
  if (!window.location.hash || !window.location.hash.startsWith('#/')) window.location.hash = '/overview';
  else renderRoute();
}

initialize();
