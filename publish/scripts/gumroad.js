/**
 * Gumroad Product Publisher
 * 
 * Usage: node gumroad.js <tool-name> <price-in-INR> <description-file>
 *   Reads credentials from ../credentials/gumroad.json
 *
 * Setup:
 * 1. Sign up at https://gumroad.com
 * 2. Settings → Advanced → Generate API access token
 * 3. Save as credentials/gumroad.json
 */

const fs = require('fs');
const path = require('path');

async function main() {
  const toolName = process.argv[2];
  const priceINR = parseInt(process.argv[3]);
  const descFile = process.argv[4];

  if (!toolName || priceINR === undefined || isNaN(priceINR)) {
    console.error('Usage: node gumroad.js <tool-name> <price-INR> [description-file]');
    console.error('Example: node gumroad.js "Color Picker Pro" 149');
    process.exit(1);
  }

  const credsPath = path.join(__dirname, '..', 'credentials', 'gumroad.json');
  if (!fs.existsSync(credsPath)) {
    console.error('Missing credentials/ gumroad.json');
    console.error('Get your token from https://gumroad.com/settings/advanced');
    process.exit(1);
  }

  const creds = JSON.parse(fs.readFileSync(credsPath, 'utf8'));
  const token = creds.accessToken || creds.token;
  if (!token) {
    console.error('gumroad.json must contain: { "accessToken": "..." }');
    process.exit(1);
  }

  let description = 'A tool from ICM Tool Factory.';
  if (descFile && fs.existsSync(descFile)) {
    description = fs.readFileSync(descFile, 'utf8');
  }

  const formData = new URLSearchParams();
  formData.append('access_token', token);
  formData.append('product[name]', toolName);
  formData.append('product[description]', description);
  formData.append('product[price_cents]', (priceINR * 100).toString());
  formData.append('product[currency]', 'INR');
  formData.append('product[require_shipping]', 'false');

  console.log(`Creating Gumroad product: ${toolName} — ₹${priceINR}...`);

  const response = await fetch('https://api.gumroad.com/v2/products', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData
  });

  const data = await response.json();

  if (data.success) {
    const p = data.product;
    console.log(`\n✅ Created: ${p.name}`);
    console.log(`   Price: ₹${p.price / 100}`);
    console.log(`   URL: ${p.short_url}`);
    console.log(`   ID: ${p.id}`);
    
    // Save the product info for later reference
    const outputPath = path.join(__dirname, '..', 'products.json');
    let products = {};
    try { products = JSON.parse(fs.readFileSync(outputPath, 'utf8')) } catch(e) {}
    products[toolName] = { id: p.id, url: p.short_url, price: p.price / 100, created: new Date().toISOString() };
    fs.writeFileSync(outputPath, JSON.stringify(products, null, 2));
    
    return p.short_url;
  } else {
    console.error('Failed:', JSON.stringify(data));
    process.exit(1);
  }
}

main().catch(e => { console.error(e); process.exit(1) });
