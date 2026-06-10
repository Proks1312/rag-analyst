# ============================================================
# run_full_pipeline.ps1
# Полный цикл: ingestion -> chunks -> load to DB -> embeddings
#
# Запуск из корня проекта:
#
#   powershell -ExecutionPolicy Bypass -File .\run_full_pipeline.ps1 `
#     -ManifestPath "data\manifests\market_reports_cable_manifest.csv"
#
# Тестовый запуск без БД и без эмбеддингов:
#
#   powershell -ExecutionPolicy Bypass -File .\run_full_pipeline.ps1 `
#     -ManifestPath "data\manifests\market_reports_cable_manifest.csv" `
#     -SkipDbLoad `
#     -SkipEmbeddings `
#     -StopOnError
#
# Опции:
#   -ManifestPath     путь к CSV-манифесту
#   -SkipIngestion    пропустить ingestion
#   -SkipChunks       пропустить сборку chunks
#   -SkipDbLoad       пропустить загрузку в Postgres
#   -SkipEmbeddings   пропустить embeddings
#   -StopOnError      остановиться при первой ошибке ingestion
# ============================================================

param(
    [string]$ManifestPath = "data\manifests\market_reports_cable_manifest.csv",
    [switch]$SkipIngestion,
    [switch]$SkipChunks,
    [switch]$SkipDbLoad,
    [switch]$SkipEmbeddings,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

# UTF-8 в консоли, чтобы кириллица не ломалась.
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$Py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Manifest = Join-Path $ProjectRoot $ManifestPath
$RagChunksDir = Join-Path $ProjectRoot "data\processed\rag_chunks"

if (-not (Test-Path $Py)) {
    Write-Host "ERROR: python venv not found: $Py" -ForegroundColor Red
    exit 1
}

function Write-Step($n, $title) {
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host "STEP $n : $title" -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

$startedAt = Get-Date

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "FULL PIPELINE START" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "Project root: $ProjectRoot"
Write-Host "Manifest:     $Manifest"
Write-Host "Python:       $Py"
Write-Host "Skip ingestion:   $SkipIngestion"
Write-Host "Skip chunks:      $SkipChunks"
Write-Host "Skip DB load:     $SkipDbLoad"
Write-Host "Skip embeddings:  $SkipEmbeddings"
Write-Host "Stop on error:    $StopOnError"
Write-Host ("=" * 80) -ForegroundColor Green

# ------------------------------------------------------------
# STEP 1 — INGESTION
# ------------------------------------------------------------
if (-not $SkipIngestion) {
    Write-Step "1/4" "Ingestion - parse PDFs from manifest"

    if (-not (Test-Path $Manifest)) {
        Write-Host "ERROR: manifest not found: $Manifest" -ForegroundColor Red

        $ManifestDir = Join-Path $ProjectRoot "data\manifests"

        if (Test-Path $ManifestDir) {
            Write-Host "Available manifests:" -ForegroundColor Yellow
            Get-ChildItem $ManifestDir -Filter "*.csv" |
                ForEach-Object {
                    Write-Host ("  " + $_.FullName) -ForegroundColor Yellow
                }
        }

        exit 1
    }

    $cmd = @(
        "run_ingestion_manifest.py",
        "--manifest", $Manifest,
        "--skip-db-load",
        "--skip-embeddings"
    )

    if ($StopOnError) {
        $cmd += "--stop-on-error"
    }

    & $Py @cmd

    if ($LASTEXITCODE -ne 0) {
        Write-Host "STEP 1 FAILED (ingestion)" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "STEP 1 skipped (-SkipIngestion)" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# STEP 2 — BUILD RAG CHUNKS
# ------------------------------------------------------------
if (-not $SkipChunks) {
    Write-Step "2/4" "Build RAG chunks from embedding records"

    & $Py build_rag_chunks_from_records.py

    if ($LASTEXITCODE -ne 0) {
        Write-Host "STEP 2 FAILED (build chunks)" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "STEP 2 skipped (-SkipChunks)" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# STEP 3 — LOAD CHUNKS TO POSTGRES
# ------------------------------------------------------------
if (-not $SkipDbLoad) {
    Write-Step "3/4" "Load RAG chunks to Postgres"

    if (-not (Test-Path $RagChunksDir)) {
        Write-Host "ERROR: rag_chunks dir not found: $RagChunksDir" -ForegroundColor Red
        exit 1
    }

    $files = Get-ChildItem $RagChunksDir -Filter "*.rag_chunks.jsonl"

    if ($files.Count -eq 0) {
        Write-Host "ERROR: no *.rag_chunks.jsonl files in $RagChunksDir" -ForegroundColor Red
        exit 1
    }

    Write-Host "Files to load: $($files.Count)"

    foreach ($f in $files) {
        Write-Host ""
        Write-Host "Loading: $($f.Name)" -ForegroundColor Yellow

        & $Py load_rag_chunks_to_db.py --rag-chunks $f.FullName

        if ($LASTEXITCODE -ne 0) {
            Write-Host "STEP 3 FAILED on $($f.Name)" -ForegroundColor Red
            exit 1
        }
    }
}
else {
    Write-Host "STEP 3 skipped (-SkipDbLoad)" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# STEP 4 — EMBEDDINGS
# ------------------------------------------------------------
if (-not $SkipEmbeddings) {
    Write-Step "4/4" "Embed chunks with BGE-M3"

    & $Py embed_chunks.py

    if ($LASTEXITCODE -ne 0) {
        Write-Host "STEP 4 FAILED (embeddings)" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "STEP 4 skipped (-SkipEmbeddings)" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------
$elapsed = (Get-Date) - $startedAt

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host "FULL PIPELINE DONE" -ForegroundColor Green
Write-Host ("Elapsed: {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Green
Write-Host ""
Write-Host "Pipeline finished." -ForegroundColor Yellow
Write-Host "Optional check:" -ForegroundColor Yellow
Write-Host ('  {0} answer_question.py test --no-llm --debug' -f $Py) -ForegroundColor Yellow
