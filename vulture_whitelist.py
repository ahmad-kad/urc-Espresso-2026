# Vulture whitelist for expected dead code
# This file contains functions, classes, and variables that are detected as unused
# but are kept for future use, API compatibility, or are used in specific contexts

# Core config schemas (kept for future schema validation)
TRAINING_SCHEMA
EVALUATION_SCHEMA

# Methods kept for API compatibility or future features
_.save_config
_.get_available_configs
_.health_check
_.warmup
_.clear_cache
_.set_model
_.set_config
_.set_results

# Pipeline infrastructure (kept for extensibility)
EvaluationError
ConversionError
DeploymentError
_._execute_with_monitoring
_.cancel
_.reset
ProgressCallback
_.on_progress
_.on_status_change
_.on_error
_.on_completion
create_pipeline_output_dir
save_pipeline_result
load_pipeline_result

# Benchmarking variables (used in context)
num_warmup_runs
num_benchmark_runs

# Utility functions kept for future use
analyze_image_sizes
ensure_device_consistency
clear_device_cache
calculate_precision_recall
calculate_map
calculate_f1_score
analyze_detection_errors
calculate_small_object_metrics

# Unused imports in error handling paths
re
diff_color

# Variables in error handling/reporting contexts
mAP95
status_str
mAP50_diff
mAP95_diff
channels
ground_truths
idx
trial_config
target_samples_per_class

# Methods kept for API compatibility
_.save_config
_.get_next

# Functions kept for future implementation
evaluate_onnx_direct

# Split variables (used in data processing context)
train_split
val_split
test_split
