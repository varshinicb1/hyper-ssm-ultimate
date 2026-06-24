async function refresh() {
  const stats = await chrome.runtime.sendMessage({ type: 'icm-get-stats' });
  if (stats?.ok) {
    document.getElementById('fact-count').textContent = stats.stats.facts;
    document.getElementById('stat-depth').textContent = stats.stats.maxDepth;
    document.getElementById('stat-nodes').textContent = stats.stats.nodes;
  }
  const all = await chrome.runtime.sendMessage({ type: 'icm-get-all' });
  const results = document.getElementById('results');
  if (!all?.ok || !all.facts.length) {
    results.innerHTML = '<div class="empty">No memories yet. Right-click text anywhere and select "Save to AI Memory".</div>';
    return;
  }
  results.innerHTML = all.facts.map(f => `
    <div class="result" data-content="${esc(f.content)}">
      <div class="topic">${esc(f.topic)}</div>
      <div class="text">${esc(f.content)}</div>
    </div>
  `).join('');
  results.querySelectorAll('.result').forEach(el => {
    el.addEventListener('click', () => {
      navigator.clipboard.writeText(el.dataset.content).then(() => {
        el.style.borderColor = '#22c55e';
        setTimeout(() => el.style.borderColor = '', 1000);
      });
    });
    el.title = 'Click to copy';
  });
}

document.getElementById('search-btn').addEventListener('click', async () => {
  const q = document.getElementById('search-input').value.trim();
  if (!q) return refresh();
  const resp = await chrome.runtime.sendMessage({ type: 'icm-query', topic: q.split(/\s+/)[0].toLowerCase(), topK: 20 });
  const results = document.getElementById('results');
  if (!resp?.ok || !resp.results.length) {
    results.innerHTML = '<div class="empty">No matching memories found.</div>'; return;
  }
  results.innerHTML = resp.results.map(r => `
    <div class="result" data-content="${esc(r.content)}">
      <div class="topic">${esc(r.topic)} <span class="sim">${(r.sim*100).toFixed(0)}%</span></div>
      <div class="text">${esc(r.content)}</div>
    </div>
  `).join('');
  results.querySelectorAll('.result').forEach(el => {
    el.addEventListener('click', () => { navigator.clipboard.writeText(el.dataset.content) });
    el.title = 'Click to copy';
  });
});

document.getElementById('search-input').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('search-btn').click() });
document.getElementById('refresh-btn').addEventListener('click', refresh);
document.getElementById('reset-btn').addEventListener('click', async () => {
  if (confirm('Clear all stored memories?')) {
    await chrome.runtime.sendMessage({ type: 'icm-reset' });
    refresh();
  }
});
refresh();

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }
