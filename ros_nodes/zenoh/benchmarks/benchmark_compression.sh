#!/bin/bash
# Unified Compression Benchmark Suite
# Tests compression levels with configurable middleware (Zenoh or CycloneDDS)
# Based on findings: <1MB frame sizes have minimal latency impact

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
VIDEO_SOURCE="${1:-0}"           # Default to camera 0, or pass video file path
DURATION="${2:-30}"               # Default 30 seconds
MIDDLEWARE="${3:-zenoh}"          # Default to zenoh, or 'cyclonedds'
OUTPUT_DIR="../output/benchmarking/compression_${MIDDLEWARE}"
RESULTS_DIR="$OUTPUT_DIR/results"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Compression Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Video Source: $VIDEO_SOURCE"
echo "Duration: ${DURATION}s per test"
echo "Middleware: $MIDDLEWARE"
echo "Output Directory: $RESULTS_DIR"
echo ""

# Validate middleware
if [ "$MIDDLEWARE" != "zenoh" ] && [ "$MIDDLEWARE" != "cyclonedds" ]; then
    echo -e "${RED}ERROR: Middleware must be 'zenoh' or 'cyclonedds'${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$RESULTS_DIR"

# Check ROS2
if ! command -v ros2 &> /dev/null; then
    echo -e "${YELLOW}Warning: ros2 command not found${NC}"
fi

source /opt/ros/jazzy/setup.bash 2>/dev/null || echo "Note: Make sure ROS2 is sourced"

# Setup middleware (scripts are in ../setup/)
if [ "$MIDDLEWARE" = "zenoh" ]; then
    source ../setup/setup_zenoh.sh
    echo -e "${GREEN}Using Zenoh middleware: $RMW_IMPLEMENTATION${NC}"
else
    source ../setup/setup_default.sh
    echo -e "${GREEN}Using CycloneDDS middleware: $RMW_IMPLEMENTATION${NC}"
fi

echo ""

# Simplified test configurations based on findings:
# - Focus on configurations that produce <1MB frames (minimal latency impact)
# - Remove redundant tests (e.g., multiple grayscale variants)
declare -a TEST_CONFIGS=(
    # Format: "name:jpeg_quality:grayscale:resolution:description"
    # Baseline
    "raw_full::0:full:Raw 640x480 (baseline)"
    # Key compression levels that keep frames <1MB
    "jpeg85_full:85:0:full:JPEG Q85, 640x480 (~18-28 Mbps)"
    "jpeg75_full:75:0:full:JPEG Q75, 640x480 (~12-18 Mbps)"
    # Reduced resolution (typically <1MB)
    "jpeg85_half:85:0:half:JPEG Q85, 320x240 (~4-6 Mbps)"
    "jpeg75_quarter:75:0:quarter:JPEG Q75, 160x120 (~0.5-1 Mbps)"
)

echo -e "${YELLOW}IMPORTANT:${NC} You need to restart camera_publisher.py for each test!"
echo "The script will prompt you before each test."
echo ""
read -p "Press Enter to start benchmarks..."

# Function to run a single benchmark
run_compression_benchmark() {
    local config_name=$1
    local jpeg_quality=$2
    local grayscale=$3
    local resolution=$4
    local description=$5
    
    local output_file="$RESULTS_DIR/${config_name}.json"
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Test: $config_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "  $description"
    echo "  JPEG Quality: ${jpeg_quality:-raw}"
    echo "  Grayscale: $([ "$grayscale" = "1" ] && echo "Yes" || echo "No")"
    echo "  Resolution: $resolution"
    echo ""
    
    echo -e "${YELLOW}Start camera publisher in another terminal:${NC}"
    ROS_NODES_DIR="$(dirname "$SCRIPT_DIR")"
    echo "  cd $ROS_NODES_DIR"
    echo "  source /opt/ros/jazzy/setup.bash"
    if [ "$MIDDLEWARE" = "zenoh" ]; then
        echo "  source setup/setup_zenoh.sh"
    else
        echo "  source setup/setup_default.sh"
    fi
    echo "  python3 camera_publisher.py --source $VIDEO_SOURCE --fps 30"
    if [ -n "$jpeg_quality" ]; then
        echo "    --jpeg-quality $jpeg_quality"
    fi
    if [ "$grayscale" = "1" ]; then
        echo "    --grayscale"
    fi
    echo "    --resolution $resolution"
    echo ""
    read -p "Press Enter when camera_publisher.py is running..."
    
    # Run benchmark (script is in current directory)
    echo "Running benchmark for ${DURATION}s..."
    python3 benchmark_camera_streaming.py \
        --topic /camera/image_raw \
        --duration "$DURATION" \
        --expected-fps 30 \
        --output "$output_file" \
        --append 2>&1 | tee /tmp/benchmark_output.log
    
    local exit_code=$?
    
    # Check results
    if [ $exit_code -eq 0 ] && [ -f "$output_file" ]; then
        # Verify valid data
        if python3 -c "import json; data=json.load(open('$output_file')); assert 'fps' in (data[-1] if isinstance(data, list) else data) and (data[-1] if isinstance(data, list) else data)['total_frames'] > 0" 2>/dev/null; then
            echo -e "${GREEN}✓ Benchmark completed: $config_name${NC}"
        else
            echo -e "${YELLOW}⚠ No valid data collected${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Benchmark had issues${NC}"
    fi
    
    echo ""
    echo "Stop camera_publisher.py (Ctrl+C) before next test"
    echo "Waiting 3 seconds..."
    sleep 3
}

# Run all benchmarks
for config in "${TEST_CONFIGS[@]}"; do
    IFS=':' read -r name jpeg_quality grayscale resolution description <<< "$config"
    run_compression_benchmark "$name" "$jpeg_quality" "$grayscale" "$resolution" "$description"
done

# Generate comparison report
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Comparison Report${NC}"
echo -e "${BLUE}========================================${NC}"

python3 << PYTHON_SCRIPT
import json
from pathlib import Path
import sys

results_dir = Path("$RESULTS_DIR")

# Find all result files
result_files = list(results_dir.glob("*.json"))
result_files = [f for f in result_files if f.name != "compression_summary.json"]

if not result_files:
    print("ERROR: No benchmark results found")
    sys.exit(1)

# Load all results
results = {}
for file in result_files:
    try:
        with open(file) as f:
            data = json.load(f)
        # Get latest result if it's a list
        if isinstance(data, list):
            result = data[-1]
        else:
            result = data
        
        config_name = file.stem
        results[config_name] = result
    except Exception as e:
        print(f"Warning: Could not load {file}: {e}")

if not results:
    print("ERROR: No valid results found")
    sys.exit(1)

# Print comparison table
print("\n" + "="*100)
print(f"COMPRESSION BENCHMARK COMPARISON ({MIDDLEWARE.upper()} Middleware)")
print("="*100)
print(f"\n{'Configuration':<25} {'FPS':<8} {'Bandwidth':<12} {'Avg Latency':<12} {'P95 Latency':<12} {'Frame Size':<12} {'Dropped':<8}")
print("-"*100)

# Sort by bandwidth (ascending)
sorted_configs = sorted(results.items(), key=lambda x: x[1].get('bandwidth_mbps', 0))

for config_name, result in sorted_configs:
    fps = result.get('fps', 0)
    bandwidth = result.get('bandwidth_mbps', 0)
    avg_latency = result.get('avg_latency_ms', 0)
    p95_latency = result.get('p95_latency_ms', 0)
    dropped = result.get('dropped_frames', 0)
    frame_size = result.get('avg_frame_size_kb', 0)
    msg_type = result.get('message_type', 'unknown')
    
    # Highlight configurations with <1MB frames (minimal latency impact)
    size_indicator = ""
    if frame_size < 1024:
        size_indicator = " <1MB"
    
    print(f"{config_name:<25} {fps:<8.2f} {bandwidth:<12.2f} {avg_latency:<12.2f} {p95_latency:<12.2f} {frame_size:<8.2f}KB{size_indicator:<4} {dropped:<8}")
    if msg_type != 'unknown':
        print(f"  └─ {msg_type}")

print("="*100)

# Calculate compression ratios (relative to raw_full)
if 'raw_full' in results:
    raw_bandwidth = results['raw_full']['bandwidth_mbps']
    raw_frame_size = results['raw_full'].get('avg_frame_size_kb', 0)
    print(f"\nCompression Ratios (relative to raw_full @ {raw_bandwidth:.2f} Mbps, {raw_frame_size:.2f} KB):")
    print("-"*100)
    print(f"{'Configuration':<25} {'Bandwidth':<12} {'Compression':<12} {'Frame Size':<12} {'Latency Impact':<20}")
    print("-"*100)
    
    raw_latency = results['raw_full'].get('avg_latency_ms', 0)
    
    for config_name, result in sorted_configs:
        if config_name == 'raw_full':
            continue
        bandwidth = result.get('bandwidth_mbps', 0)
        compression_ratio = raw_bandwidth / bandwidth if bandwidth > 0 else 0
        frame_size = result.get('avg_frame_size_kb', 0)
        latency = result.get('avg_latency_ms', 0)
        latency_diff = latency - raw_latency
        latency_impact = f"{latency_diff:+.2f}ms"
        if frame_size < 1024:
            latency_impact += " (minimal)"
        
        print(f"{config_name:<25} {bandwidth:<12.2f} {compression_ratio:<12.2f}x {frame_size:<8.2f}KB{'':<4} {latency_impact:<20}")

# Save summary
summary = {
    'middleware': '$MIDDLEWARE',
    'raw_bandwidth_mbps': raw_bandwidth if 'raw_full' in results else 0,
    'raw_frame_size_kb': raw_frame_size if 'raw_full' in results else 0,
    'results': {name: {
        'fps': r.get('fps', 0),
        'bandwidth_mbps': r.get('bandwidth_mbps', 0),
        'avg_latency_ms': r.get('avg_latency_ms', 0),
        'p95_latency_ms': r.get('p95_latency_ms', 0),
        'dropped_frames': r.get('dropped_frames', 0),
        'avg_frame_size_kb': r.get('avg_frame_size_kb', 0),
        'compression_ratio': raw_bandwidth / r.get('bandwidth_mbps', 1) if 'raw_full' in results and r.get('bandwidth_mbps', 0) > 0 else 0,
        'latency_impact_ms': r.get('avg_latency_ms', 0) - raw_latency if 'raw_full' in results else 0,
        'message_type': r.get('message_type', 'unknown')
    } for name, r in results.items()}
}

summary_file = results_dir / "compression_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_file.absolute()}")
print("="*100)
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Compression Benchmark Suite Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "  - Individual test results: *.json"
echo "  - Summary: compression_summary.json"
echo ""
echo "Key Finding: Configurations with <1MB frame sizes show minimal latency impact"
echo ""

