param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$RepositoryName = 'inference-engine-textbook'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed (exit $LASTEXITCODE). No repository visibility change will be attempted."
    }
}

# Fetch only the account login; never print or store an authentication token.
$account = & gh api user --jq '.login'
if ($LASTEXITCODE -ne 0 -or -not $account) {
    throw 'GitHub CLI cannot reach an authenticated account. Check network access and gh auth status, then retry.'
}
$account = $account.Trim()
$repository = "$account/$RepositoryName"

Invoke-Checked -Executable python -Arguments @('scripts/package_pages.py')
if (-not (Test-Path -LiteralPath '.git')) {
    Invoke-Checked -Executable git -Arguments @('init', '-b', 'main')
}
$branch = & git branch --show-current
if ($LASTEXITCODE -ne 0 -or $branch.Trim() -ne 'main') {
    throw 'This publisher requires the main branch. Review the current Git checkout before retrying.'
}
$remotes = @(& git remote)
if ($LASTEXITCODE -ne 0) {
    throw 'Could not inspect Git remotes.'
}
if ($remotes -contains 'origin') {
    $origin = & git remote get-url origin
    throw "An origin remote already exists ($origin). This script only creates a new private repository; review and push the existing remote manually."
}
Invoke-Checked -Executable git -Arguments @('add', '--all')
$changes = & git diff --cached --name-only
if ($changes) {
    Invoke-Checked -Executable git -Arguments @('commit', '-m', 'Publish inference engine research textbook and Pages workflow')
}

Invoke-Checked -Executable gh -Arguments @('repo', 'create', $repository, '--private', '--source', '.', '--remote', 'origin', '--push', '--description', 'Research textbook on inference engines, from first principles to serving systems')
$privateFlag = & gh repo view $repository --json isPrivate --jq '.isPrivate'
if ($LASTEXITCODE -ne 0 -or $privateFlag.Trim() -ne 'true') {
    throw 'Could not verify repository privacy. Stopping before configuring Pages.'
}

# GitHub Pages may be public even though the repository is private.
# A paid plan is required for private personal-repository Pages.
& gh api --method POST "repos/$repository/pages" -f build_type=workflow
if ($LASTEXITCODE -ne 0) {
    throw "The private repository was uploaded, but GitHub rejected Pages setup. Check plan eligibility and repository policy at https://github.com/$repository/settings/pages. The repository remains private."
}
Invoke-Checked -Executable gh -Arguments @('workflow', 'run', 'pages.yml', '--repo', $repository, '--ref', 'main')
Write-Host "Private repository: https://github.com/$repository"
Write-Host "Deployment runs: https://github.com/$repository/actions/workflows/pages.yml"
& gh api "repos/$repository/pages" --jq '.html_url'

