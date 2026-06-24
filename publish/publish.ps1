#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Unified publish command for ICM Tool Factory.
  Dispatches to platform-specific scripts based on tool type.

.PARAMETER ToolPath
  Path to the tool directory (e.g. tools/chrome-extensions/color-picker)

.PARAMETER Platform
  Target platform: chrome-webstore, gumroad, github-pages, all

.PARAMETER Price
  Price in INR (for gumroad)

.EXAMPLE
  .\publish.ps1 tools\chrome-extensions\color-picker -Platform chrome-webstore
  .\publish.ps1 tools\chrome-extensions\ai-memory -Platform all -Price 299
#>

param(
  [Parameter(Mandatory=$true)]
  [string]$ToolPath,
  [Parameter(Mandatory=$false)]
  [ValidateSet('chrome-webstore','gumroad','github-pages','all','deploy-web')]
  [string]$Platform = 'all',
  [int]$Price = 0
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolPath = Resolve-Path (Join-Path $RepoRoot $ToolPath) -ErrorAction Stop
$ToolName = Split-Path $ToolPath -Leaf

# Determine tool type from path
$IsChromeExt = $ToolPath -match 'chrome-extensions'
$IsWebTool = $ToolPath -match 'web-tools'
$IsVsCodeExt = $ToolPath -match 'vscode-extensions'

function Publish-ChromeWebStore {
  Write-Host "🚀 Publishing Chrome extension: $ToolName" -ForegroundColor Cyan
  node "$PSScriptRoot\scripts\chrome-upload.js" "$ToolPath"
  if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Chrome extension published" -ForegroundColor Green
  } else {
    Write-Host "❌ Chrome extension publish failed" -ForegroundColor Red
  }
}

function Publish-Gumroad {
  if ($Price -le 0) {
    Write-Host "⚠️  Skipping Gumroad: no price specified. Use -Price <INR>" -ForegroundColor Yellow
    return
  }
  Write-Host "🚀 Publishing to Gumroad: $ToolName — ₹$Price" -ForegroundColor Cyan
  node "$PSScriptRoot\scripts\gumroad.js" "$ToolName" $Price
  if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Gumroad product created" -ForegroundColor Green
  } else {
    Write-Host "❌ Gumroad publish failed" -ForegroundColor Red
  }
}

function Publish-GitHubPages {
  Write-Host "🚀 Deploying to GitHub Pages: $ToolName" -ForegroundColor Cyan
  # Web tools are deployed by committing to main branch
  # GitHub Actions picks up the changes automatically
  git add "$ToolPath"
  git add "tools/index.html"
  git commit -m "chore: publish $ToolName [automated]"
  git push
  if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Pushed to GitHub. Pages will deploy in 1-2 min." -ForegroundColor Green
    Write-Host "   URL: https://varshinicb1.github.io/hyper-ssm-ultimate/tools/" -ForegroundColor Cyan
  } else {
    Write-Host "❌ Git push failed" -ForegroundColor Red
  }
}

function Publish-DeployWeb {
  Write-Host "🚀 Deploying web tool: $ToolName" -ForegroundColor Cyan
  git add "$ToolPath"
  git commit -m "chore: deploy $ToolName web tool [automated]"
  git push
  if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Deployed to GitHub Pages" -ForegroundColor Green
  }
}

# Dispatch
switch ($Platform) {
  'chrome-webstore' { Publish-ChromeWebStore }
  'gumroad' { Publish-Gumroad }
  'github-pages' { Publish-GitHubPages }
  'deploy-web' { Publish-DeployWeb }
  'all' {
    if ($IsChromeExt) {
      Publish-ChromeWebStore
      if ($Price -gt 0) { Publish-Gumroad }
    }
    elseif ($IsWebTool) {
      Publish-DeployWeb
    }
    Write-Host "`n✅ All done!" -ForegroundColor Green
  }
}
