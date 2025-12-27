# PowerShell script to train all 4 models in parallel terminals
# Each model runs in its own terminal window

Write-Host "Starting parallel training for all 4 confidence-optimized models..." -ForegroundColor Green
Write-Host "Each model will open in a separate terminal window" -ForegroundColor Yellow
Write-Host ""

# Start YOLOv8s Base in Terminal 1
Write-Host "Starting YOLOv8s Base training..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList "/k yolo train cfg=""configs/training/yolov8s_confidence.yaml"" workers=4 cache=disk && pause" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start YOLOv8s CBAM in Terminal 2
Write-Host "Starting YOLOv8s CBAM training..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList "/k yolo train cfg=""configs/training/yolov8s_cbam_confidence.yaml"" workers=4 cache=disk && pause" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start MobileNet-ViT in Terminal 3
Write-Host "Starting MobileNet-ViT training..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList "/k yolo train cfg=""configs/training/mobilenet_confidence.yaml"" workers=4 cache=disk && pause" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start EfficientNet in Terminal 4
Write-Host "Starting EfficientNet training..." -ForegroundColor Cyan
Start-Process -FilePath "cmd.exe" -ArgumentList "/k yolo train cfg=""configs/training/efficientnet_confidence.yaml"" workers=4 cache=disk && pause" -WindowStyle Normal

Write-Host ""
Write-Host "All 4 models are now training in parallel!" -ForegroundColor Green
Write-Host "Monitor each terminal window for progress" -ForegroundColor Yellow
Write-Host "Training results will be saved to: output/confidence/" -ForegroundColor Yellow
Write-Host ""
Write-Host "GPU Acceleration: ENABLED" -ForegroundColor Magenta
Write-Host "Mixed Precision (AMP): ENABLED" -ForegroundColor Magenta
Write-Host "Batch Sizes: 16-24 (optimized per architecture)" -ForegroundColor Magenta
Write-Host "Workers: 4 (optimized for Windows multiprocessing)" -ForegroundColor Magenta
Write-Host "Cache: disk (deterministic, stable training)" -ForegroundColor Magenta
Write-Host "Epochs: 100" -ForegroundColor Magenta
Write-Host "Multiscale: ENABLED" -ForegroundColor Magenta
Write-Host "Early Stopping: ENABLED" -ForegroundColor Magenta
