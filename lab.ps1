param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Rest
)

& "$PSScriptRoot\.venv\Scripts\python.exe" -m tracker.lab @Rest
exit $LASTEXITCODE
