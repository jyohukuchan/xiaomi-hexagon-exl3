function Invoke-ProjectAdbQuery {
    param(
        [Parameter(Mandatory)][string]$Adb,
        [Parameter(Mandatory)][string]$Serial,
        [Parameter(Mandatory)][string]$Command,
        [ValidateRange(100, 60000)][int]$TimeoutMs = 10000
    )
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Adb
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in @('-s', $Serial, 'shell', $Command)) { $startInfo.ArgumentList.Add($argument) }
    $query = [System.Diagnostics.Process]::new()
    $query.StartInfo = $startInfo
    try {
        if (-not $query.Start()) { throw 'Could not start the ADB query' }
        $outputTask = $query.StandardOutput.ReadToEndAsync()
        $errorTask = $query.StandardError.ReadToEndAsync()
        if (-not $query.WaitForExit($TimeoutMs)) {
            $query.Kill()
            $null = $query.WaitForExit(1000)
            throw "ADB query timed out after $TimeoutMs ms; device process state is unknown"
        }
        $queryOutput = $outputTask.GetAwaiter().GetResult()
        $queryError = $errorTask.GetAwaiter().GetResult()
        if ($query.ExitCode -ne 0) { throw "ADB query failed: $queryError" }
        return $queryOutput.Trim()
    } finally {
        $query.Dispose()
    }
}
