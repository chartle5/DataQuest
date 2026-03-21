document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('demoForm');
  const trainBtn = document.getElementById('trainBtn');
  const rankBtn = document.getElementById('rankBtn');
  const message = document.getElementById('message');
  const trialInfo = document.getElementById('trialInfo');
  const resultsDiv = document.getElementById('results');
  const metricsPanel = document.getElementById('metricsPanel');

  // --- Rank candidates ---
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const condition = document.getElementById('conditionSelect').value;
    const topN = parseInt(document.getElementById('topN').value, 10) || 5;
    const conditionLabel = document.getElementById('conditionSelect').selectedOptions[0].text;

    showMessage(`Ranking top ${topN} candidates for <strong>${conditionLabel}</strong>...`, 'info');
    rankBtn.disabled = true;
    rankBtn.textContent = 'Ranking...';

    fetch('/api/rank', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition: condition, top_n: topN })
    })
      .then(r => r.json())
      .then(data => {
        rankBtn.disabled = false;
        rankBtn.textContent = 'Rank Candidates';

        if (data.error) {
          showMessage(data.error, 'error');
          return;
        }

        // Show trial info
        if (data.trial_id) {
          trialInfo.innerHTML = `<div class="trial-badge">Trial: <strong>${escapeHtml(data.trial_id)}</strong></div>` +
            (data.trial_title ? `<div class="trial-title">${escapeHtml(data.trial_title)}</div>` : '') +
            (data.output_file ? `<div class="output-note">Output saved to: <code>${escapeHtml(data.output_file)}</code></div>` : '');
          trialInfo.classList.add('visible');
        }

        showMessage(`Found ${data.candidates.length} candidates`, 'success');
        renderResults(data.candidates);
        fetchMetrics();
      })
      .catch(err => {
        rankBtn.disabled = false;
        rankBtn.textContent = 'Rank Candidates';
        showMessage('Failed to reach server. Is it running on port 8080?', 'error');
      });
  });

  // --- Train model ---
  trainBtn.addEventListener('click', function () {
    const condition = document.getElementById('conditionSelect').value;
    const conditionLabel = document.getElementById('conditionSelect').selectedOptions[0].text;

    showMessage(`Training model for <strong>${conditionLabel}</strong>... this may take a moment.`, 'info');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';

    fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition: condition })
    })
      .then(r => r.json())
      .then(data => {
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Model';
        if (data.status === 'trained') {
          showMessage(`Model trained successfully! Path: <code>${data.model_path || 'N/A'}</code>`, 'success');
        } else {
          showMessage('Training completed with warnings.', 'info');
        }
      })
      .catch(err => {
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Model';
        showMessage('Training request failed. Check the server logs.', 'error');
      });
  });

  function fetchMetrics() {
    fetch('/api/metrics')
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => {
        if (data.error) return;
        let html = '<h3>Model Evaluation Metrics</h3><div class="metrics-grid">';
        for (const [key, val] of Object.entries(data)) {
          html += `<div class="metric"><span class="metric-label">${escapeHtml(key)}</span><span class="metric-value">${typeof val === 'number' ? val.toFixed(4) : val}</span></div>`;
        }
        html += '</div>';
        metricsPanel.innerHTML = html;
        metricsPanel.classList.add('visible');
      })
      .catch(() => { /* no metrics yet */ });
  }

  function showMessage(html, type) {
    message.innerHTML = html;
    message.className = 'message visible ' + type;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function renderResults(candidates) {
    resultsDiv.innerHTML = '';
    if (!candidates || candidates.length === 0) {
      resultsDiv.innerHTML = '<p>No candidates found.</p>';
      return;
    }

    const table = document.createElement('table');
    table.className = 'results-table';

    // Header
    const thead = document.createElement('thead');
    thead.innerHTML = `<tr>
      <th>#</th><th>Patient</th><th>Score</th><th>Confidence</th><th>Reasons</th>
    </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    candidates.forEach((c, i) => {
      const tr = document.createElement('tr');
      const confidenceClass = c.confidence === 'high' ? 'conf-high' : c.confidence === 'uncertain' ? 'conf-uncertain' : c.confidence === 'ineligible' ? 'conf-low' : 'conf-moderate';
      const nameDisplay = (c.first || c.last) ? `${c.first} ${c.last}` : c.patient_id.substring(0, 12) + '...';

      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>
          <div class="patient-name">${escapeHtml(nameDisplay)}</div>
          <div class="patient-id">${escapeHtml(c.patient_id.substring(0, 20))}...</div>
        </td>
        <td><span class="score-badge">${c.score.toFixed(4)}</span></td>
        <td><span class="confidence-badge ${confidenceClass}">${escapeHtml(c.confidence)}</span></td>
        <td><ul class="reason-list">${(c.reasons || []).map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul></td>
      `;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    resultsDiv.appendChild(table);
  }

  // Load metrics on page load (if available)
  fetchMetrics();
});
