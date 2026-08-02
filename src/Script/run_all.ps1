$exePath = "D:\Users\YYLee\cb_cpp\build_trace\convertible_bond.exe"
$marketData = "D:\Users\YYLee\cb_cpp\res_table.csv"
$tickers = @("FSLY", "VERI", "COLL", "DHR", "EEFT", "LAB", "SPOT", "EXPE")
$tickers = @("SPOT", "EXPE", "EEFT", "LAB")

$origianlTitle = $Host.UI.RawUI.WindowTitle


Write-Host "Starting automated crash hunting..." -ForegroundColor Cyan

foreach ($ticker in $tickers) {
    $currentTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Host.UI.RawUI.WindowTitle = "GDB Debugging - Processing: $ticker - $currentTime"
    # -batch: Runs silently without the GDB welcome text
    # -ex "run": Automatically starts the program
    # -ex "bt": Prints the stack trace IF it crashes
    & gdb -batch -ex "run" -ex "bt" --args $exePath $ticker $marketData
}

$Host.UI.RawUI.WindowTitle = $origianlTitle

Write-Host "`nBatch complete." -ForegroundColor Green