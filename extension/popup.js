// ── Config ────────────────────────────────────────────────────────────────
const API_BASE = 'https://flashpoint.watch';

// ── Helpers ───────────────────────────────────────────────────────────────
function scoreColor(s) {
  return s >= 70 ? '#22c55e' : s >= 40 ? '#f59e0b' : '#ef4444';
}
function scoreLabel(s) {
  return s >= 70 ? 'Credible' : s >= 40 ? 'Mixed' : 'Low Credibility';
}

// ── UI states ─────────────────────────────────────────────────────────────
function showLoading(msg = 'Analyzing credibility…') {
  document.getElementById('state').innerHTML = `
    <div class="loading-bar"></div>
    <div class="loading-text">${msg}</div>
  `;
}

function showError(msg) {
  document.getElementById('state').innerHTML = `
    <div class="error-box">${msg}</div>
  `;
}

function renderResults(data) {
  const score = data.overall_score;
  const clr   = scoreColor(score);
  const lbl   = scoreLabel(score);

  const dims = (data.dimensions || []).map(d => {
    const dc = scoreColor(d.score);
    return `
      <div class="dim">
        <div class="dim-top">
          <span class="dim-name">${d.name}</span>
          <span class="dim-score" style="color:${dc}">${d.score}</span>
        </div>
        <div class="dim-bar-bg">
          <div class="dim-bar" style="width:${d.score}%;background:${dc}"></div>
        </div>
      </div>`;
  }).join('');

  const shareUrl = data.share_id
    ? `${API_BASE}/pramaana?id=${data.share_id}`
    : `${API_BASE}/pramaana`;

  document.getElementById('state').innerHTML = `
    <div class="score-row">
      <div class="score-num" style="color:${clr}">${score}</div>
      <div class="score-right">
        <div class="score-lbl" style="background:${clr}22;color:${clr};border:1px solid ${clr}44">${lbl}</div>
        <div class="outlet">${data.outlet || ''}</div>
      </div>
    </div>

    <div class="art-title">${data.article_title || ''}</div>
    <div class="verdict">${data.verdict || ''}</div>

    <div class="dims-section">
      <div class="sec-label">Credibility Dimensions</div>
      <div class="dims">${dims}</div>
    </div>

    <div class="funding-section">
      <div class="sec-label">Source &amp; Funding</div>
      <div class="funding-text">${data.source_funding || ''}</div>
    </div>

    <div class="actions">
      <a href="${shareUrl}" target="_blank" class="btn-view">View full analysis →</a>
      ${data.share_id ? `<button class="btn-copy" id="copy-btn">Copy link</button>` : ''}
    </div>
  `;

  // Copy link handler
  if (data.share_id) {
    document.getElementById('copy-btn').addEventListener('click', function () {
      navigator.clipboard.writeText(shareUrl).then(() => {
        this.textContent = 'Copied!';
        this.style.color = '#22c55e';
        setTimeout(() => {
          this.textContent = 'Copy link';
          this.style.color = '';
        }, 2000);
      });
    });
  }
}

// ── Core analyze ──────────────────────────────────────────────────────────
function analyze(url) {
  showLoading('Analyzing credibility…');
  fetch(`${API_BASE}/api/pramaana/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
    .then(r => {
      if (!r.ok) throw new Error(`Server error ${r.status}`);
      return r.json();
    })
    .then(data => {
      if (data.error) throw new Error(data.error);
      renderResults(data);
    })
    .catch(err => showError(err.message || 'Analysis failed. Try again.'));
}

// ── Init ──────────────────────────────────────────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const url = tabs[0]?.url || '';

  // Guard against non-article pages
  if (
    !url ||
    url.startsWith('chrome://') ||
    url.startsWith('chrome-extension://') ||
    url.startsWith('about:') ||
    url.startsWith('edge://')
  ) {
    showError("Navigate to a news article first, then click the extension.");
    return;
  }

  analyze(url);
});
