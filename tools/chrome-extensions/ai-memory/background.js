importScripts('lib/memory-tree.js');
const STORAGE_KEY = 'icm_memory_tree';

let tree = null;

async function loadTree() {
  const data = (await chrome.storage.local.get(STORAGE_KEY))[STORAGE_KEY];
  if (data) {
    try { tree = HyperbolicMemoryTree.fromJSON(JSON.parse(data)); } catch (e) { console.error('ICM: Failed to load tree', e); tree = new HyperbolicMemoryTree(); }
  } else {
    tree = new HyperbolicMemoryTree();
  }
}

async function saveTree() {
  await chrome.storage.local.set({ [STORAGE_KEY]: JSON.stringify(tree.toJSON()) });
}

chrome.runtime.onInstalled.addListener(async () => {
  await loadTree();
  chrome.contextMenus.create({
    id: 'icm-save-selection',
    title: 'Save to AI Memory',
    contexts: ['selection']
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'icm-save-selection') {
    await loadTree();
    const topic = guessTopic(info.selectionText);
    const emb = seededEmb(topic);
    tree.remember(emb, info.selectionText, topic);
    await saveTree();
    chrome.tabs.sendMessage(tab.id, { type: 'icm-toast', msg: `Saved to memory: "${info.selectionText.slice(0, 50)}..."` }).catch(() => { });
  }
});

chrome.runtime.onMessage.addListener(async (msg, sender, sendResponse) => {
  await loadTree();
  switch (msg.type) {
    case 'icm-save': {
      const emb = seededEmb(msg.topic);
      tree.remember(emb, msg.content, msg.topic);
      await saveTree();
      sendResponse({ ok: true, stats: tree.getStats() });
      break;
    }
    case 'icm-recall': {
      const emb = seededEmb(msg.topic);
      const results = tree.recall(emb, msg.topK || 5);
      sendResponse({ ok: true, results });
      break;
    }
    case 'icm-query': {
      const emb = seededEmb(msg.topic);
      const results = tree.recall(emb, msg.topK || 5);
      sendResponse({ ok: true, results });
      break;
    }
    case 'icm-get-stats': {
      sendResponse({ ok: true, stats: tree.getStats() });
      break;
    }
    case 'icm-get-all': {
      sendResponse({ ok: true, facts: tree.getAllFacts() });
      break;
    }
    case 'icm-reset': {
      tree.reset();
      await saveTree();
      sendResponse({ ok: true });
      break;
    }
  }
  return true;
});

function guessTopic(text) {
  const common = ['the','a','an','is','are','was','were','my','your','his','her','its','our','their','in','on','at','to','for','of','with','by','from','and','or','but','this','that','it','be','been','being','have','has','had','do','does','did','will','would','could','should','may','might','shall','can','not','no','nor','so','if','as'];
  const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(w => w.length > 2 && !common.includes(w));
  return words.length ? words[0] : text.slice(0, 10);
}
