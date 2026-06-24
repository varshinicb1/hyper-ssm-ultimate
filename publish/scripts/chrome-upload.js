/**
 * Chrome Web Store Publisher
 * 
 * Usage: node chrome-upload.js <extension-path>
 *   Reads credentials from ../credentials/chrome-store.json
 *   Requires: Google Cloud service account with Chrome Web Store API enabled
 *
 * Setup:
 * 1. Go to https://console.cloud.google.com → Create project
 * 2. Enable Chrome Web Store API
 * 3. Create service account → Download JSON key
 * 4. Go to https://chrome.google.com/webstore/devconsole
 * 5. Upload extension manually ONCE → note the item ID
 * 6. Grant service account email access in dev console
 * 7. Save credentials as credentials/chrome-store.json
 */

const fs = require('fs');
const path = require('path');

async function main() {
  const extPath = process.argv[2];
  if (!extPath) { console.error('Usage: node chrome-upload.js <extension-path>'); process.exit(1) }

  const credsPath = path.join(__dirname, '..', 'credentials', 'chrome-store.json');
  if (!fs.existsSync(credsPath)) {
    console.error('Missing credentials file. See scripts/README.md for setup.');
    process.exit(1);
  }

  const creds = JSON.parse(fs.readFileSync(credsPath, 'utf8'));

  // Load chrome-webstore-upload (install: npm install chrome-webstore-upload)
  let chromeWebstoreUpload;
  try {
    chromeWebstoreUpload = require('chrome-webstore-upload');
  } catch(e) {
    console.error('Install chrome-webstore-upload first:');
    console.error('  cd publish && npm install chrome-webstore-upload');
    process.exit(1);
  }

  const store = chromeWebstoreUpload.default({
    extensionId: creds.itemId,
    clientId: creds.clientId,
    clientSecret: creds.clientSecret,
    refreshToken: creds.refreshToken
  });

  // Read the extension zip
  const zipName = path.basename(extPath) + '.zip';
  const zipPath = path.join(extPath, zipName);
  if (!fs.existsSync(zipPath)) {
    // Try to find any .zip in the directory
    const files = fs.readdirSync(extPath).filter(f => f.endsWith('.zip'));
    if (files.length === 0) {
      console.error(`No .zip found in ${extPath}. Create one first: compress the extension folder.`);
      process.exit(1);
    }
    zipPath = path.join(extPath, files[0]);
  }

  const zipStream = fs.createReadStream(zipPath);
  const token = await store.fetchToken();
  
  console.log('Uploading to Chrome Web Store...');
  const uploadRes = await store.uploadExisting(zipStream, token);
  console.log('Upload response:', JSON.stringify(uploadRes));

  if (uploadRes.uploadState === 'FAILURE') {
    console.error('Upload failed:', uploadRes.itemError);
    process.exit(1);
  }

  console.log('Publishing...');
  const publishRes = await store.publish('default', token);
  console.log('Publish response:', JSON.stringify(publishRes));

  console.log(`\n✅ Published successfully!`);
  console.log(`   https://chrome.google.com/webstore/detail/${creds.itemId}`);
}

main().catch(e => { console.error(e); process.exit(1) });
