const defaultResponse = "The Eiffel Tower is in Paris. OpenAI was founded in 2015. The Great Wall of China is a series of fortifications across northern China.";
const API_BASE_URL = document.querySelector('meta[name="verillm-api-base"]')?.content.trim() || window.location.origin;

const summaryFields = {
  totalEvaluations: document.getElementById('summary-total-evaluations'),
  claimsAnalyzed: document.getElementById('summary-claims-analyzed'),
  supported: document.getElementById('metric-supported'),
  contradicted: document.getElementById('metric-contradicted'),
  unsupported: document.getElementById('metric-unsupported'),
  highSeverity: document.getElementById('metric-high-severity'),
  humanReview: document.getElementById('metric-human-review'),
  evidence: document.getElementById('metric-evidence'),
  supportRate: document.getElementById('support-rate'),
  mode: document.getElementById('verification-mode')
};

function toTitleCase(value) {
  return String(value || '').replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function updateSummary(summary = {}, claims = []) {
  const total = Number(summary.total_claims || 0);
  const supported = Number(summary.supported || 0);
  const contradicted = Number(summary.contradicted || 0);
  const unsupported = Number(summary.unsupported || 0);
  const highSeverity = Number(summary.high_severity_count || 0);
  const humanReview = Number(summary.human_review_count || 0);
  const supportRate = total ? Math.round((supported / total) * 100) : 0;

  summaryFields.totalEvaluations.textContent = total ? '1' : '0';
  summaryFields.claimsAnalyzed.textContent = total;
  summaryFields.supported.textContent = supported;
  summaryFields.contradicted.textContent = contradicted;
  summaryFields.unsupported.textContent = unsupported;
  summaryFields.highSeverity.textContent = highSeverity;
  summaryFields.humanReview.textContent = humanReview;
  summaryFields.evidence.textContent = claims.reduce((count, claim) => count + (Array.isArray(claim.evidence) ? claim.evidence.length : 0), 0) || 0;
  summaryFields.supportRate.textContent = `${supportRate}%`;

  const totalBars = Math.max(total, 1);
  document.getElementById('bar-supported').style.width = `${(supported / totalBars) * 100}%`;
  document.getElementById('bar-contradicted').style.width = `${(contradicted / totalBars) * 100}%`;
  document.getElementById('bar-unsupported').style.width = `${(unsupported / totalBars) * 100}%`;
  document.getElementById('bar-supported-label').textContent = supported;
  document.getElementById('bar-contradicted-label').textContent = contradicted;
  document.getElementById('bar-unsupported-label').textContent = unsupported;

  const severityMap = {
    LOW: 'sev-low',
    MEDIUM: 'sev-medium',
    HIGH: 'sev-high',
    CRITICAL: 'sev-critical'
  };

  Object.keys(severityMap).forEach((level) => {
    const tag = severityMap[level];
    const count = claims.filter((claim) => claim.severity === level).length;
    const width = total ? (count / total) * 100 : 0;
    document.getElementById(tag).style.width = `${width}%`;
    document.getElementById(`${tag}-label`).textContent = count;
  });

  const ring = document.querySelector('.score-ring');
  ring.style.background = `conic-gradient(var(--sky-strong) 0deg ${supportRate * 3.6}deg, rgba(155,201,240,0.15) ${supportRate * 3.6}deg 360deg)`;
}

function renderClaims(claims) {
  const container = document.getElementById('claims-container');
  container.innerHTML = '';

  if (!claims.length) {
    container.innerHTML = '<div class="claim-item"><p>No claims were extracted from this response.</p></div>';
    return;
  }

  claims.forEach((claim) => {
    const item = document.createElement('article');
    item.className = 'claim-item';
    const verdictClass = `verdict-${String(claim.verdict || 'unsupported').toLowerCase()}`;
    item.innerHTML = `
      <div class="claim-header">
        <span class="claim-title">${claim.claim_id || 'Claim'}</span>
        <span class="verdict-badge ${verdictClass}">${String(claim.verdict || 'Unsupported').toUpperCase()}</span>
      </div>
      <div class="claim-meta">
        <span>Confidence: ${(Number(claim.confidence || 0) * 100).toFixed(0)}%</span>
        <span>Severity: ${claim.severity || 'LOW'}</span>
        <span>Retrieval: ${(Number(claim.retrieval_score || 0) * 100).toFixed(0)}%</span>
      </div>
      <p class="claim-text">${claim.claim || 'No claim text available.'}</p>
      <div class="claim-meta">
        <span>Type: ${toTitleCase(claim.hallucination_type || 'Unknown')}</span>
        <span>${claim.needs_human_review ? 'Human review required' : 'Auto-verified'}</span>
      </div>
    `;
    container.appendChild(item);
  });
}

function renderEvidence(claims) {
  const container = document.getElementById('evidence-container');
  const items = claims.flatMap((claim) => {
    const evidence = Array.isArray(claim.evidence) ? claim.evidence : [];
    return evidence.map((entry) => ({
      claim: claim.claim,
      text: entry.evidence || entry,
      score: Number(entry.score || 0),
      verdict: claim.verdict
    }));
  });

  container.innerHTML = '';

  if (!items.length) {
    container.innerHTML = '<div class="evidence-item"><p>No evidence was retrieved for the current response.</p></div>';
    return;
  }

  items.forEach((entry) => {
    const item = document.createElement('article');
    item.className = 'evidence-item';
    item.innerHTML = `
      <div class="claim-header">
        <span class="claim-title">${toTitleCase(entry.verdict)}</span>
        <span class="verdict-badge verdict-${entry.verdict.toLowerCase()}">${(entry.score * 100).toFixed(0)}%</span>
      </div>
      <p>${entry.text}</p>
    `;
    container.appendChild(item);
  });
}

async function fetchAnalysis(responseText) {
  const response = await fetch(`${API_BASE_URL}/api/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ response: responseText })
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || 'Verification failed.');
  }

  return response.json();
}

async function loadOverview() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/overview`);
    if (!response.ok) {
      throw new Error('Overview endpoint failed.');
    }
    const data = await response.json();
    const claims = data.claims || [];
    updateSummary(data.summary || {}, claims);
    renderClaims(claims);
    renderEvidence(claims);
    summaryFields.mode.textContent = data.mode || 'ready';
  } catch (error) {
    summaryFields.mode.textContent = 'offline';
    console.error(error);
  }
}

const form = document.getElementById('verify-form');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const responseText = document.getElementById('response-input').value.trim();
  if (!responseText) {
    return;
  }

  summaryFields.mode.textContent = 'running';
  try {
    const result = await fetchAnalysis(responseText);
    const claims = result.claims || [];
    updateSummary(result.summary || {}, claims);
    renderClaims(claims);
    renderEvidence(claims);
    summaryFields.mode.textContent = result.mode || 'ready';
  } catch (error) {
    summaryFields.mode.textContent = 'error';
    alert(error.message);
  }
});

document.getElementById('load-demo').addEventListener('click', () => {
  document.getElementById('response-input').value = defaultResponse;
  form.requestSubmit();
});

loadOverview();
