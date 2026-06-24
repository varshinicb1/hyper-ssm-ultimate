// ─── Inject ICM panel into ChatGPT / Claude pages ───
(function () {
  const HOST = window.location.hostname;
  const IS_CHATGPT = HOST.includes('chatgpt.com') || HOST.includes('chat.openai.com');
  const IS_CLAUDE = HOST.includes('claude.ai');
  const IS_GEMINI = HOST.includes('gemini.google.com');
  if (!IS_CHATGPT && !IS_CLAUDE && !IS_GEMINI) return;
  const PLATFORM = IS_CHATGPT ? 'ChatGPT' : IS_CLAUDE ? 'Claude' : 'Gemini';

  const container = document.createElement('div');
  container.id = 'icm-memory-panel';
  container.innerHTML = `
    <div id="icm-toggle" title="AI Memory">🧠 <span id="icm-badge">0</span></div>
    <div id="icm-panel">
      <div id="icm-header">
        <span>AI Memory</span>
        <span id="icm-close">✕</span>
      </div>
      <div id="icm-body">
        <div id="icm-save-row">
          <input id="icm-topic-input" placeholder="Topic (e.g. project, preference, code)">
          <button id="icm-save-btn">Save</button>
        </div>
        <div id="icm-search-row">
          <input id="icm-search-input" placeholder="Search memories...">
          <button id="icm-search-btn">Search</button>
        </div>
        <div id="icm-results"></div>
        <div id="icm-stats">
          <span id="icm-fact-count">0</span> memories stored
        </div>
        <button id="icm-inject-btn" style="display:none">Inject Selected</button>
      </div>
    </div>
  `;
  document.body.appendChild(container);

  const toggle = container.querySelector('#icm-toggle');
  const panel = container.querySelector('#icm-panel');
  const close = container.querySelector('#icm-close');
  const topicInput = container.querySelector('#icm-topic-input');
  const saveBtn = container.querySelector('#icm-save-btn');
  const searchInput = container.querySelector('#icm-search-input');
  const searchBtn = container.querySelector('#icm-search-btn');
  const results = container.querySelector('#icm-results');
  const badge = container.querySelector('#icm-badge');
  const factCount = container.querySelector('#icm-fact-count');
  const injectBtn = container.querySelector('#icm-inject-btn');

  let selectedResults = [];
  let isOpen = false;

  toggle.addEventListener('click', () => {
    isOpen = !isOpen; panel.style.display = isOpen ? 'flex' : 'none';
    if (isOpen) updateStats();
  });
  close.addEventListener('click', () => { isOpen = false; panel.style.display = 'none' });

  function toast(msg) {
    let t = container.querySelector('#icm-toast');
    if (!t) { t = document.createElement('div'); t.id = 'icm-toast'; container.appendChild(t) }
    t.textContent = msg; t.style.display = 'block'; t.style.opacity = '1';
    clearTimeout(t._t); t._t = setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.style.display = 'none', 300) }, 2500);
  }

  async function updateStats() {
    const resp = await chrome.runtime.sendMessage({ type: 'icm-get-stats' });
    if (resp?.ok) { badge.textContent = resp.stats.facts; factCount.textContent = resp.stats.facts }
  }

  // Save selected text from ChatGPT
  async function saveMemory(content, topic) {
    if (!topic) { const w = content.split(/\s+/).filter(w => w.length > 3); topic = w.length ? w[0] : 'general' }
    const resp = await chrome.runtime.sendMessage({ type: 'icm-save', content, topic: topic.toLowerCase() });
    if (resp?.ok) { toast('Saved!'); updateStats() }
  }

  saveBtn.addEventListener('click', async () => {
    const selection = window.getSelection().toString().trim();
    if (!selection) { toast('Select text on the page first'); return }
    const topic = topicInput.value.trim() || guessTopic(selection);
    await saveMemory(selection, topic);
    topicInput.value = '';
  });

  // Quick save: listen for clicks on message text
  document.addEventListener('dblclick', async (e) => {
    const sel = window.getSelection().toString().trim();
    if (sel.length > 10 && isOpen) {
      topicInput.value = guessTopic(sel);
      topicInput.focus();
    }
  });

  searchBtn.addEventListener('click', async () => {
    const q = searchInput.value.trim();
    if (!q) return;
    const resp = await chrome.runtime.sendMessage({ type: 'icm-query', topic: guessTopic(q), topK: 10 });
    if (!resp?.ok) return;
    selectedResults = resp.results || [];
    renderResults(selectedResults, q);
  });

  searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') searchBtn.click() });
  topicInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') saveBtn.click() });

  function renderResults(items, query) {
    if (!items.length) { results.innerHTML = '<div class="icm-empty">No matching memories found.</div>'; injectBtn.style.display = 'none'; return }
    let html = items.map((r, i) => `
      <div class="icm-result" data-idx="${i}" data-content="${escapeHtml(r.content)}" data-topic="${escapeHtml(r.topic)}">
        <div class="icm-r-topic">${escapeHtml(r.topic)}</div>
        <div class="icm-r-text">${escapeHtml(r.content)}</div>
        <div class="icm-r-sim">${(r.sim * 100).toFixed(1)}% match</div>
      </div>
    `).join('');
    results.innerHTML = html;
    injectBtn.style.display = 'block';

    results.querySelectorAll('.icm-result').forEach(el => {
      el.addEventListener('click', () => {
        el.classList.toggle('icm-selected');
        const idx = parseInt(el.dataset.idx);
        selectedResults[idx]._selected = el.classList.contains('icm-selected');
      });
    });
  }

  injectBtn.addEventListener('click', () => {
    const selected = selectedResults.filter(r => r._selected);
    const text = selected.map(r => `[Memory: ${r.topic}] ${r.content}`).join('\n\n');
    if (!text) { toast('Select memories by clicking them'); return }

    // Try to inject into ChatGPT input
    const input = document.querySelector('textarea, [contenteditable="true"], [role="textbox"]');
    if (input) {
      if (input.tagName === 'TEXTAREA' || input.tagName === 'INPUT') {
        input.value = 'From my memories:\n' + text + '\n\n' + input.value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      } else if (input.isContentEditable) {
        input.focus();
        document.execCommand('insertText', false, 'From my memories:\n' + text + '\n\n');
      }
      toast('Injected ' + selected.length + ' memories into chat');
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(text).then(() => toast('Copied to clipboard! Paste into chat.')).catch(() => toast('Select the chat input first'));
    }
  });

  // Listen for background messages
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'icm-toast') toast(msg.msg);
  });

  updateStats();

  function escapeHtml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;') }
  function guessTopic(text) {
    const common = ['the','a','an','is','are','was','were','my','your','his','her','its','our','their','in','on','at','to','for','of','with','by','from','and','or','but','this','that','it','not','no','so','if','as','be','been','being','have','has','had'];
    const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(w => w.length > 2 && !common.includes(w));
    return words.length ? words[0] : text.slice(0, 10);
  }

  // ─── Auto-suggest: memories appear as you type ───
  const suggestEl = document.createElement('div');
  suggestEl.id = 'icm-suggest';
  container.appendChild(suggestEl);

  let suggestTimer = null;
  function watchInput() {
    const input = document.querySelector('#prompt-textarea, [role="textbox"] textarea, textarea');
    if (!input) { setTimeout(watchInput, 1000); return }
    input.addEventListener('input', () => {
      clearTimeout(suggestTimer);
      suggestTimer = setTimeout(async () => {
        const text = input.value || input.textContent || '';
        const words = text.trim().split(/\s+/).filter(w => w.length > 2);
        if (words.length < 2) { suggestEl.style.display = 'none'; return }
        const topic = guessTopic(text);
        const resp = await chrome.runtime.sendMessage({ type: 'icm-query', topic, topK: 3 });
        if (!resp?.ok || !resp.results.length) { suggestEl.style.display = 'none'; return }
        const items = resp.results.filter(r => r.sim > 0.4);
        if (!items.length) { suggestEl.style.display = 'none'; return }
        showSuggestions(items);
      }, 600);
    });
  }

  function showSuggestions(items) {
    suggestEl.innerHTML = items.map(r => `
      <div class="icm-sug-item" data-content="${escapeHtml(r.content)}">
        <span class="icm-sug-topic">${escapeHtml(r.topic)}</span>
        <span class="icm-sug-text">${escapeHtml(r.content.slice(0, 80))}</span>
      </div>
    `).join('');
    suggestEl.style.display = 'block';
    suggestEl.querySelectorAll('.icm-sug-item').forEach(el => {
      el.addEventListener('click', () => {
        const text = el.dataset.content;
        const input = document.querySelector('#prompt-textarea, [role="textbox"] textarea, textarea');
        if (input) {
          if (input.isContentEditable) {
            input.focus();
            document.execCommand('insertText', false, text + '\n\n');
          } else {
            const start = input.selectionStart || 0;
            input.value = input.value.slice(0, start) + text + '\n\n' + input.value.slice(input.selectionEnd || 0);
            input.dispatchEvent(new Event('input', { bubbles: true }));
          }
        }
        suggestEl.style.display = 'none';
        toast('Memory injected');
      });
    });
  }

  watchInput();

  // ─── Auto-save: watch ChatGPT for new messages ───
  // Use sessionStorage so dedup survives page refreshes within the tab
  let savedHashes;
  try { savedHashes = new Set(JSON.parse(sessionStorage.getItem('icm_saved') || '[]')) } catch(e) { savedHashes = new Set() }
  function persistHashes() { try { sessionStorage.setItem('icm_saved', JSON.stringify([...savedHashes])) } catch(e) {} }

  function hashText(s) { let h = 0; for (let i = 0; i < s.length; i++) { h = ((h << 5) - h) + s.charCodeAt(i); h |= 0 } return Math.abs(h).toString(16) }

  function extractMessageText(el) {
    // ChatGPT uses data-message-author-role, Claude uses .message
    // Walk DOM to find actual text content (skip buttons, menus, etc.)
    const clone = el.cloneNode(true);
    clone.querySelectorAll('button, svg, [role="button"], .sr-only, .visually-hidden').forEach(n => n.remove());
    return clone.textContent.replace(/\s+/g, ' ').trim();
  }

  async function autoSaveMessage(el, role) {
    const text = extractMessageText(el);
    if (text.length < 20) return; // skip too-short messages
    const h = hashText(text);
    if (savedHashes.has(h)) return; // already saved
    savedHashes.add(h); persistHashes();

    const topic = role === 'user' ? guessTopic(text) : 'ai-' + guessTopic(text);
    await chrome.runtime.sendMessage({ type: 'icm-save', content: text.slice(0, 500), topic });
    updateStats();

    // Add a subtle indicator
    const indicator = document.createElement('span');
    indicator.className = 'icm-auto-saved';
    indicator.title = 'Saved to AI Memory';
    indicator.textContent = '✓';
    Object.assign(indicator.style, {
      fontSize: '10px', color: '#22c55e', marginLeft: '6px', opacity: '0.6',
      fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
      userSelect: 'none', pointerEvents: 'none'
    });
    const header = el.querySelector('[data-message-author-role]') || el.querySelector('.message-role') || el;
    if (!el.querySelector('.icm-auto-saved')) {
      // Find a good place: last text block or header
      const target = el.querySelector('header, [role="heading"], .font-token') || el.querySelector('h1, h2, h3, h4, h5, h6') || el;
      target.appendChild(indicator);
    }
  }

  function startAutoSave() {
    let seen = new Set();
    // Watch for new message elements
    const obs = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType !== 1) continue;
          const el = node;
          // ChatGPT style: data-message-author-role
          if (el.matches && el.matches('[data-message-author-role]')) {
            const role = el.getAttribute('data-message-author-role');
            autoSaveMessage(el, role);
            continue;
          }
          // Check children for messages
          const userMsg = el.querySelector ? el.querySelector('[data-message-author-role="user"]') : null;
          if (userMsg) { autoSaveMessage(userMsg, 'user') }
          const asstMsg = el.querySelector ? el.querySelector('[data-message-author-role="assistant"]') : null;
          if (asstMsg) { autoSaveMessage(asstMsg, 'assistant') }
        }
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
    // Also scan existing messages on load
    setTimeout(() => {
      document.querySelectorAll('[data-message-author-role]').forEach(el => {
        const role = el.getAttribute('data-message-author-role');
        autoSaveMessage(el, role);
      });
    }, 2000);
  }

  if (IS_CHATGPT) startAutoSave();
})();
