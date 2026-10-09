$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$submodule = Join-Path $repo 'third_party\llama.cpp'
$patch = Join-Path $repo 'patches\llama-exl3.patch'
git -C $submodule apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -eq 0) { exit 0 }
git -C $submodule apply --check $patch
if ($LASTEXITCODE -ne 0) { throw 'The runtime patch conflicts with the current submodule. Preserve local edits and resolve the conflict.' }
git -C $submodule apply $patch
if ($LASTEXITCODE -ne 0) { throw 'Could not apply the runtime patch' }
