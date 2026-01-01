#  Maintenance Guide

**Guidelines for maintaining and updating the YOLO AI Camera system**

##  Repository Structure

```
urc-espresso-2026/
├──  deployment_package/     #  PRODUCTION deployment system
│   ├──  models/            # Optimized ONNX models
│   ├──  scripts/           # Production inference scripts
│   ├──  config/            # Alert rules & service config
│   ├──  docs/              # Deployment documentation
│   └──  tools/             # Analysis & benchmarking tools
│
├──  scripts/               #  Maintenance & training scripts
│   ├── convert_to_onnx.py    # Model conversion utilities
│   ├── benchmark_models.py   # Performance benchmarking
│   ├── train_all_models.bat  # Windows training script
│   └── retrain_confidence_cpu.sh # CPU retraining script
│
├──  utils/                 #  Utility modules
│   ├── data_utils.py         # Data processing utilities
│   ├── metrics.py            # Performance metrics
│   └── visualization.py      # Plotting and visualization
│
├──  configs/               #  Model configurations
├──  output/                #  Training outputs & results
├──  docs/                  #  Documentation
├──  setup_dev.py           # Development environment setup
├── .gitignore                # Git ignore rules
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── pyrightconfig.json        # Pyright type checking config
├── .flake8                   # Flake8 linting configuration
├── .mypy.ini                 # MyPy type checking configuration
└──  consolidated_dataset/  #  Training dataset
```

##  Development Tools

### Code Quality Setup
```bash
# Install development tools
python setup_dev.py

# Or manually install
pip install black isort flake8 mypy pre-commit
pre-commit install
```

### Code Quality Checks
```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type check
mypy .

# Run all pre-commit checks
pre-commit run --all-files
```

### IDE Configuration
- **VS Code**: Install Python extension, Pyright will use `pyrightconfig.json`
- **PyCharm**: Enable mypy integration, configure Black as formatter
- **Other editors**: Use pre-commit hooks for consistent formatting

##  Update Procedures

### Model Updates

1. **Retrain Model** (if needed for better accuracy):
   ```bash
   # Use existing training scripts
   python scripts/train_all_models_complete.py

   # Or retrain specific model
   yolo train model=yolov8n.pt data=consolidated_dataset/data.yaml \
     epochs=100 imgsz=224 batch=16 device=cpu
   ```

2. **Convert to ONNX**:
   ```bash
   python scripts/convert_to_onnx.py --model_path output/models/yolov8n_fixed_224/weights/best.pt
   ```

3. **Update Deployment Package**:
   ```bash
   # Copy new model to deployment package
   cp output/models/*/weights/best.onnx deployment_package/models/

   # Update deployment README if needed
   # Test deployment scripts
   ```

### Performance Monitoring

1. **Benchmark Current Model**:
   ```bash
   python scripts/benchmark_models.py
   ```

2. **Per-Class Accuracy Check**:
   ```bash
   python scripts/evaluate_per_class_accuracy.py
   ```

3. **Model Comparison**:
   ```bash
   python scripts/evaluation/run_comparison.py
   ```

### Dataset Updates

1. **Add New Training Data**:
   - Add images to `consolidated_dataset/train/images/`
   - Add annotations to `consolidated_dataset/train/labels/`
   - Update class labels if needed

2. **Validate Dataset**:
   ```bash
   python scripts/dataset_utils.py --validate
   ```

##  Deployment Updates

### Update Production System

1. **Test Locally**:
   ```bash
   # Test inference scripts
   python deployment_package/scripts/run_inference_optimized.py

   # Test alert system
   python deployment_package/scripts/alert_manager.py
   ```

2. **Update Deployment Scripts**:
   - Modify `deployment_package/deploy_to_pi.sh` if needed
   - Update `deployment_package/install.sh` for new dependencies
   - Test service configuration

3. **Deploy to Production**:
   ```bash
   # Deploy updated system
   ./deployment_package/deploy_to_pi.sh raspberrypi.local

   # Verify deployment
   ssh pi@raspberrypi.local "python3 ~/yolo-detector/scripts/health_check.py"
   ```

### Configuration Updates

1. **Alert Rules**: Edit `deployment_package/config/alert_config.json`
2. **Service Settings**: Edit `deployment_package/config/service_config.json`
3. **Model Thresholds**: Update confidence/IoU values based on benchmarking

##  Troubleshooting

### Common Issues

**Model Performance Degradation**:
```bash
# Re-run accuracy analysis
   python scripts/evaluate_per_class_accuracy.py

# Check threshold tuning
   python scripts/evaluation/run_comparison.py
```

**Deployment Failures**:
```bash
# Check Raspberry Pi connectivity
ping raspberrypi.local

# Verify model file integrity
ls -la deployment_package/models/
```

**Service Issues**:
```bash
# On Raspberry Pi
sudo systemctl status yolo-detector
sudo journalctl -u yolo-detector -n 50
```

### Performance Optimization

**If FPS drops below 15**:
1. Check system resources: `htop`
2. Review model size vs performance trade-offs
3. Consider model quantization
4. Update to latest ONNX Runtime

**If accuracy drops**:
1. Re-tune confidence/IoU thresholds
2. Check for dataset drift
3. Consider model retraining

##  Monitoring & Alerts

### Key Metrics to Monitor

- **System Health**: CPU, memory, temperature
- **Detection Performance**: FPS, detection counts
- **Accuracy**: False positive/negative rates
- **Service Uptime**: Systemd service status

### Alert Configuration

Current alert rules in `deployment_package/config/alert_config.json`:
- **Bottle Detection**: Alerts when bottles are detected
- **Hammer Count**: Alerts when >2 hammers detected
- **ArUco Persistence**: Alerts when ArUco tags present >5 seconds
- **Zone Intrusion**: Alerts when hammers enter restricted zones

##  Security Considerations

### Repository Security
- Never commit model files >100MB (use Git LFS if needed)
- Keep credentials out of repository
- Use .gitignore for sensitive files

### Deployment Security
- Service runs as `pi` user (not root)
- No new privileges allowed
- Private temp directories
- Resource limits enforced

##  Version Control

### Release Process
1. Update version in relevant files
2. Run full test suite
3. Update deployment package
4. Create release tag
5. Update documentation

### Branching Strategy
- `main`: Production-ready code
- `development`: New features and updates
- `hotfix-*`: Critical bug fixes

##  Support

### Getting Help
1. Check `docs/` for detailed documentation
2. Review `deployment_package/README.md` for deployment issues
3. Check service logs: `sudo journalctl -u yolo-detector`
4. Run diagnostics: `python3 scripts/health_check.py`

### Performance Baselines
- **Expected FPS**: 200-250 FPS (ONNX on Raspberry Pi 5)
- **Accuracy**: 83% F1-Score overall
- **Memory Usage**: <400MB during operation
- **CPU Usage**: <80% sustained load

---

**Remember**: Always test updates in a development environment before deploying to production!
stained load

---

**Remember**: Always test updates in a development environment before deploying to production!
o production!
