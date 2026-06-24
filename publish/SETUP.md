# Publishing Platform Setup Guide

Complete the steps below once. After that, I can publish tools directly.

---

## ✅ GitHub Pages (Already Working)

Web tools at `tools/web-tools/*` auto-deploy when I commit. You just need to enable Pages:

1. Repo → Settings → Pages
2. Source: **GitHub Actions**
3. Done. Your tools live at:
   - `https://varshinicb1.github.io/hyper-ssm-ultimate/tools/web-tools/json-formatter/`
   - `https://varshinicb1.github.io/hyper-ssm-ultimate/tools/web-tools/markdown-editor/`

---

## 🖥️ Chrome Web Store

I need these 4 values to publish extensions automatically:

### Step 1: Create Google Cloud Project (5 min)
1. Go to https://console.cloud.google.com
2. Create a new project → name it "ICM Tool Factory"
3. Go to **APIs & Services** → **Library** → search "Chrome Web Store API" → Enable

### Step 2: Create OAuth Credentials (5 min)
1. In Google Cloud, go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth client ID**
3. Application type: **Desktop app**
4. Name: "ICM Publisher"
5. Download the JSON → save as `publish/credentials/chrome-store-oauth.json`
6. Note the **Client ID** and **Client Secret** from this file

### Step 3: Get Refresh Token (5 min)
1. Open this URL (replace YOUR_CLIENT_ID):
   ```
   https://accounts.google.com/o/oauth2/auth?response_type=code&scope=https://www.googleapis.com/auth/chromewebstore&client_id=YOUR_CLIENT_ID&redirect_uri=urn:ietf:wg:oauth:2.0:oob
   ```
2. Sign in → copy the authorization code
3. Exchange it for tokens:
   ```bash
   curl -d "code=AUTHORIZATION_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET&redirect_uri=urn:ietf:wg:oauth:2.0:oob&grant_type=authorization_code" https://oauth2.googleapis.com/token
   ```
4. Copy the `refresh_token` from the response

### Step 4: Upload First Extension Manually (5 min)
1. Open https://chrome.google.com/webstore/devconsole
2. Click **New item** → upload any `.zip`
3. Fill in basic info (you can edit later)
4. Save as draft → note the **Item ID** from the URL

### Step 5: Grant Service Account Access (2 min)
1. In the dev console, go to your item → **Account** tab
2. Add your email as a user with **Owner** role

### Step 6: Save Credentials

Save to `publish/credentials/chrome-store.json`:
```json
{
  "itemId": "THE_ITEM_ID_FROM_STEP_4",
  "clientId": "YOUR_CLIENT_ID",
  "clientSecret": "YOUR_CLIENT_SECRET",
  "refreshToken": "YOUR_REFRESH_TOKEN"
}
```

✅ After this, I can publish by running:
```
.\publish\publish.ps1 tools\chrome-extensions\color-picker -Platform chrome-webstore
```

---

## 📦 Gumroad

### Step 1: Create Account (5 min)
1. Sign up at https://gumroad.com
2. Go to **Settings** → **Payments**
3. Add **Razorpay** as payment processor (supports UPI, cards, net banking)

### Step 2: Get API Token (2 min)
1. Go to **Settings** → **Advanced**
2. Scroll to **API Access Token**
3. Click **Generate**
4. Copy the token

### Step 3: Save Credentials

Save to `publish/credentials/gumroad.json`:
```json
{
  "accessToken": "YOUR_GUMROAD_ACCESS_TOKEN"
}
```

✅ After this, I can create products by running:
```
.\publish\publish.ps1 tools\chrome-extensions\color-picker -Platform gumroad -Price 149
```

---

## 🏦 Razorpay (Payment Links)

### Step 1: Create Account (10 min)
1. Sign up at https://razorpay.com
2. Complete your business verification (PAN, bank account, etc.)

### Step 2: Get API Keys (2 min)
1. Go to **Settings** → **API Keys**
2. Generate a new key pair
3. Copy **Key ID** and **Key Secret**

### Step 3: Save Credentials

Add to `publish/credentials/.env`:
```
RAZORPAY_KEY_ID=rzp_live_xxxxxxxx
RAZORPAY_KEY_SECRET=your-key-secret
```

---

## 🤖 After Setup: How I'll Publish

Once credentials are saved, you just tell me:

> "Publish the color picker extension"

And I'll run:
1. Pack the extension → `.\publish\publish.ps1 tools\chrome-extensions\color-picker -Platform all`
2. Web Store upload → done
3. Gumroad product → done
4. GitHub commit → done

Revenue lands in your:
- **Bank account** (via Razorpay/Gumroad payouts)
- **Google Pay** (via Chrome Web Store payouts)
