let currentColor = '#4A7CFF';

function rgb2hex(r,g,b) { return '#' + [r,g,b].map(x => x.toString(16).padStart(2,'0')).join('').toUpperCase() }
function hex2rgb(h) { const r = parseInt(h.slice(1,3),16), g = parseInt(h.slice(3,5),16), b = parseInt(h.slice(5,7),16); return [r,g,b] }
function rgb2hsl(r,g,b) { r/=255; g/=255; b/=255; const M=Math.max(r,g,b), m=Math.min(r,g,b), d=M-m; let h=0,s=0,l=(M+m)/2; if(d){s=l>0.5?d/(2-M-m):d/(M+m); if(M===r)h=((g-b)/d)%6; else if(M===g)h=(b-r)/d+2; else h=(r-g)/d+4; h=Math.round(h*60); if(h<0)h+=360} return [Math.round(h), Math.round(s*100), Math.round(l*100)] }
function rgb2cmyk(r,g,b) { const k=1-Math.max(r/255,g/255,b/255); if(k===1)return[0,0,0,100]; return [Math.round((1-r/255-k)/(1-k)*100),Math.round((1-g/255-k)/(1-k)*100),Math.round((1-b/255-k)/(1-k)*100),Math.round(k*100)] }

function updateUI(color) {
  currentColor = color;
  const [r,g,b] = hex2rgb(color);
  const [h,s,l] = rgb2hsl(r,g,b);
  const [c,m,y,k] = rgb2cmyk(r,g,b);
  document.getElementById('preview').style.background = color;
  document.getElementById('preview').textContent = color;
  document.getElementById('f-hex').textContent = color;
  document.getElementById('f-rgb').textContent = `rgb(${r},${g},${b})`;
  document.getElementById('f-hsl').textContent = `hsl(${h},${s}%,${l}%)`;
  document.getElementById('f-cmyk').textContent = `cmyk(${c}%,${m}%,${y}%,${k}%)`;
  const textColor = (r*0.299 + g*0.587 + b*0.114) > 128 ? '#000' : '#fff';
  document.getElementById('preview').style.color = textColor;
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    document.getElementById('preview').textContent = 'Copied!';
    setTimeout(() => updateUI(currentColor), 600);
  });
}

document.querySelectorAll('.format').forEach(el => {
  el.addEventListener('click', () => copyText(el.textContent));
});
document.getElementById('copy-btn').addEventListener('click', () => copyText(currentColor));

document.getElementById('random-btn').addEventListener('click', () => {
  const c = '#' + Array.from({length:6},()=>'0123456789ABCDEF'[Math.floor(Math.random()*16)]).join('');
  updateUI(c); saveColor(c);
});

document.getElementById('pick-btn').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({active:true, currentWindow:true});
  try {
    await chrome.scripting.executeScript({ target: {tabId: tab.id}, files: ['content.js'] });
    chrome.tabs.sendMessage(tab.id, { type: 'activate-picker' });
  } catch(e) {
    // Fallback: generate random color
    const c = '#' + Array.from({length:6},()=>'0123456789ABCDEF'[Math.floor(Math.random()*16)]).join('');
    updateUI(c); saveColor(c);
  }
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'color-picked' && msg.color) {
    updateUI(msg.color); saveColor(msg.color);
  }
});

async function saveColor(c) {
  const {colors=[]} = await chrome.storage.local.get('colors');
  if (!colors.includes(c)) { colors.unshift(c); if (colors.length > 20) colors.pop() }
  await chrome.storage.local.set({colors});
  renderHistory();
}

async function renderHistory() {
  const {colors=[]} = await chrome.storage.local.get('colors');
  document.getElementById('h-count').textContent = colors.length;
  const h = document.getElementById('history');
  h.innerHTML = colors.map(c => `<div class="swatch" style="background:${c}" data-color="${c}" title="${c}"></div>`).join('');
  h.querySelectorAll('.swatch').forEach(el => {
    el.addEventListener('click', () => { updateUI(el.dataset.color); copyText(el.dataset.color) });
  });
}

document.getElementById('clear-btn').addEventListener('click', async (e) => {
  e.preventDefault();
  await chrome.storage.local.set({colors:[]});
  renderHistory();
});

renderHistory();
