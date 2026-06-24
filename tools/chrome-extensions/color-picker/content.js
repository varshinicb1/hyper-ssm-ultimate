// Color Picker Pro — uses EyeDropper API (Chrome 95+)
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'activate-picker') {
    try {
      if (!window.EyeDropper) { sendResponse({ ok: false, error: 'EyeDropper API not supported' }); return }
      const ed = new EyeDropper();
      ed.open().then(result => {
        chrome.runtime.sendMessage({ type: 'color-picked', color: result.sRGBHex });
        sendResponse({ ok: true, color: result.sRGBHex });
      }).catch(() => { sendResponse({ ok: false }) });
    } catch(e) { sendResponse({ ok: false, error: e.message }) }
    return true;
  }
});
