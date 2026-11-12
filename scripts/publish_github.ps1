# Publish the extracted full project directory using Git for Windows.
# Requires an existing EMPTY repository on GitHub and a Git login/PAT/SSH configuration.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is not installed. Install Git for Windows first.'
}
$url = Read-Host 'Enter your empty GitHub repository URL (https://github.com/USERNAME/REPO.git)'
if (-not ($url -match '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$' -or $url -match '^git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$')) {
    throw 'Invalid GitHub repository URL.'
}
if (-not (Test-Path .git)) { git init -b main }
git add .
git diff --cached --stat
$confirm = Read-Host 'Commit and push these files? (yes/no)'
if ($confirm -ne 'yes') { Write-Host 'Canceled.'; exit 0 }
git commit -m 'Release Arabic Sign Language Recognition V5 project'
if (-not ((git remote) -contains 'origin')) { git remote add origin $url }
git branch -M main
git push -u origin main
