# Publishing Guide — ICM Tool Factory

## How to publish and earn money from each tool type.

---

## 1. Chrome Extensions (₹99–₹299 one-time)

### Setup (one-time)
1. Create a **Chrome Web Store Developer account**: https://chrome.google.com/webstore/devconsole
2. Pay **$5 (≈₹415)** registration fee — one-time, lifetime access

### Publish each extension
1. Open `chrome://extensions` → Enable **Developer mode**
2. Click **Pack extension** → select the extension folder (e.g. `tools/chrome-extensions/ai-memory/`)
3. This creates a `.crx` (packed extension) + `.pem` (private key — **keep this safe**)
4. Go to [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole)
5. Click **New item** → upload the `.crx` file
6. Fill in:
   - **Description** (use `web-store-listing.md` in each extension folder)
   - **Screenshots** (1280×800px — use the `screenshot-mockup.html` tool)
   - **Icon** (128×128px — already in each extension)
   - **Pricing**: Set as paid (₹99–₹299)
7. Google takes **30% commission**, you get **70%**
8. Payout via Google Pay to your Indian bank account

### Update an extension
1. Increment version in `manifest.json`
2. Repack with the same `.pem` key
3. Upload new `.crx` to dashboard

### Extensions ready to publish
| Extension | Price | Status |
|-----------|-------|--------|
| AI Memory | ₹299 | Ready (in `chrome-extensions/ai-memory/`) |
| Color Picker Pro | ₹149 | Ready (in `chrome-extensions/color-picker/`) |

---

## 2. Web Tools (Free + donations)

### Setup (one-time)
1. These are pure HTML/JS files → host on **GitHub Pages** for free
2. Enable GitHub Pages on your repo: Settings → Pages → Source → GitHub Actions
3. Or use any static hosting (Netlify, Vercel — free tiers)

### Monetize
- **Buy Me a Coffee** links (already included)
- **GitHub Sponsors** links (already included)
- Sell premium version via **Gumroad** (works with Razorpay for India)

### Gumroad Setup (for premium tools)
1. Sign up at [gumroad.com](https://gumroad.com) — free
2. Go to Settings → Payments → add Razorpay (supports UPI, cards, net banking)
3. Create a product → set price in INR
4. Get the product link → add "Buy Premium" button to your free tool

### Web tools ready
| Tool | Price | Status |
|------|-------|--------|
| JSON Formatter & Validator | Free | Ready (`web-tools/json-formatter/`) |
| Markdown Editor | Free | Ready (`web-tools/markdown-editor/`) |

---

## 3. VS Code Extensions (Free, donation-based)

### Setup
1. Create a publisher account: https://marketplace.visualstudio.com/vscode
2. Install `vsce` CLI: `npm install -g @vscode/vsce`
3. Login: `vsce login <publisher-name>`

### Publish
1. Navigate to extension folder
2. Run `vsce package` to create `.vsix` file
3. Run `vsce publish` to publish to marketplace
4. Or upload `.vsix` manually in the marketplace dashboard

### Monetize
- VS Code Marketplace doesn't support paid extensions
- Add **Buy Me a Coffee** and **GitHub Sponsors** links in the extension README

---

## Revenue Summary

| Stream | Per-sale | Monthly potential | Effort |
|--------|----------|-------------------|--------|
| Chrome Extensions (₹149-₹299) | ~₹105-₹209 after Google's 30% | ₹5K-₹50K (20-200 sales) | Medium |
| Gumroad Premium (₹99-₹499) | ~₹89-₹449 after 10% + fees | ₹2K-₹20K | Low |
| Donations (GitHub Sponsors) | ₹300-₹3,000/mo | ₹3K-₹30K | Low |
| Buy Me a Coffee | ₹30-₹300 | ₹1K-₹10K | Very low |

### Total monthly potential: **₹11K – ₹1.1L / month**

---

## AI Agent Automation

The AI agent (opencode) can:
- ✅ Build new extensions/tools from scratch
- ✅ Generate ZIP packages ready to upload
- ✅ Write Chrome Web Store listings
- ✅ Create Gumroad product descriptions
- ✅ Bump version numbers for updates
- ❌ Cannot upload to stores (needs your account login)
- ❌ Cannot process payments (handled by the platform)

**Workflow**: You say "build me [tool idea]" → AI builds it → You publish → Money flows.
