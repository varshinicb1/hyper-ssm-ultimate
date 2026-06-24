# AI Memory — Chrome Extension

Adds infinite context memory to ChatGPT, Claude, and Gemini.

## How to Install

1. Open Chrome and go to `chrome://extensions`
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select this folder (`chrome-extension/`)

## How to Use

**Save:** Right-click any text on ChatGPT/Claude → "Save to AI Memory"

**Recall:** Click the 🧠 floating button → Search → Select → Inject

**Popup:** Click the extension icon in your toolbar to browse all memories.

**Auto-suggest:** As you type in ChatGPT, related memories appear automatically.

## Files

| File | Purpose |
|------|---------|
| `manifest.json` | Extension config |
| `background.js` | Service worker — manages memory tree, context menu |
| `lib/memory-tree.js` | HyperbolicMemoryTree — tree-structured recall |
| `content/content.js` | Injects memory panel on ChatGPT/Claude pages |
| `content/content.css` | Styling for injected elements |
| `popup/popup.html` | Extension popup for browsing/searching memories |
| `popup/popup.js` | Popup logic |
| `icons/` | Extension icons |

## Privacy

All memories are stored locally in your browser via `chrome.storage.local`. No data leaves your device.
