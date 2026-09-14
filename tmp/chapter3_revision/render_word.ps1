param([string]$InputDocument, [string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$chapterWord = New-Object -ComObject Word.Application
$chapterWord.Visible = $false
$chapterWord.DisplayAlerts = 0
try {
    $chapterDoc = $chapterWord.Documents.Open($InputDocument, $false, $true)
    $chapterDoc.Repaginate()
    $chapterDoc.ExportAsFixedFormat($OutputPdf, 17)
    Write-Output ('Pages: ' + $chapterDoc.ComputeStatistics(2))
    $chapterDoc.Close(0)
} finally {
    $chapterWord.Quit()
}
