document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('demoForm');
  const message = document.getElementById('message');

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const conditionEl = document.getElementById('conditionSelect');
    const topNEl = document.getElementById('topN');
    const conditionText = conditionEl.options[conditionEl.selectedIndex].text;
    const topN = topNEl.value || '5';

    message.innerHTML = `<strong>${conditionText}</strong> — top ${topN} candidates requested. Backend coming soon.`;
    message.classList.add('visible');
  });
});
