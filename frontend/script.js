document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('demoForm');
  const message = document.getElementById('message');

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const conditionEl = document.getElementById('conditionSelect');
    const topNEl = document.getElementById('topN');
    const conditionText = conditionEl.options[conditionEl.selectedIndex].text;
    const topN = topNEl.value || '5';

    message.innerHTML = `<strong>${conditionText}</strong> — top ${topN} candidates requested.`;
    message.classList.add('visible');

    // Call backend API to fetch ranked candidates
    // map frontend selection to backend condition query
    const conditionQuery = conditionEl.value === 'type2' ? 'type 2 diabetes' : (conditionEl.value === 'cancer' ? 'cancer' : conditionEl.value);

    fetch('/api/rank', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition: conditionQuery, top_n: parseInt(topN, 10) })
    }).then(r => r.json()).then(data => {
      if (data.error) {
        results.innerHTML = '<p class="error">'+data.error+'</p>';
        return;
      }
      const candidates = (data.candidates || []).map(c => ({ id: c.patient_id, first: c.first || '', last: c.last || '', score: c.score, rationale: (c.reasons || []).join('; ') }));
      // If server returns only ids and scores, display ids
      renderResults(candidates);
    }).catch(err => {
      // fallback to demo
      const demoCandidates = [
        {id: '72a36ff8-...', first: 'Jarred', last: 'Hudson', score: 0.95, rationale: 'age within required range; condition matches; completeness=1.00'},
        {id: '9074c69b-...', first: 'Alvaro', last: 'Haverhill', score: 0.92, rationale: 'age within required range; condition matches; completeness=1.00'}
      ];
      renderResults(demoCandidates);
    });
  });
});


function renderResults(candidates) {
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '';
  if (!candidates || candidates.length === 0) {
    resultsDiv.innerHTML = '<p>No candidates found.</p>';
    return;
  }

  const list = document.createElement('ol');
  candidates.forEach(c => {
    const item = document.createElement('li');
    item.innerHTML = `<strong>${c.first} ${c.last}</strong> (<code>${c.id}</code>) — score: ${c.score.toFixed(2)}<br/><em>${c.rationale}</em>`;
    list.appendChild(item);
  });
  resultsDiv.appendChild(list);
}
