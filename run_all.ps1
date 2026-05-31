$exePath = "D:\Users\YYLee\cb_cpp\build_trace\convertible_bond.exe"
$marketData = "D:\Users\YYLee\cb_cpp\res_table.csv"
$tickers = @("EXPE", "FSLY", "VERI", "SPOT", "DHR", "COLL", "EEFT", "LAB")
$tickers = @("COLL")


Write-Host "Starting automated crash hunting..." -ForegroundColor Cyan

foreach ($ticker in $tickers) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host " Running Ticker: $ticker" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    # -batch: Runs silently without the GDB welcome text
    # -ex "run": Automatically starts the program
    # -ex "bt": Prints the stack trace IF it crashes
    & gdb -batch -ex "run" -ex "bt" --args $exePath $ticker $marketData
}

Write-Host "`nBatch complete." -ForegroundColor Green