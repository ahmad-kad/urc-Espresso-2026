#!/bin/bash
# Unified Middleware Benchmark Runner
# Compares Zenoh vs CycloneDDS middleware performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VIDEO_SOURCE="${1:-0}"  # Default to camera 0, or pass video file path
DURATION="${2:-30}"     # Default 30 seconds

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Middleware Benchmark Comparison${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Video Source: $VIDEO_SOURCE"
echo "Duration: ${DURATION}s per middleware"
echo ""

source /opt/ros/jazzy/setup.bash 2>/dev/null || echo "Note: Make sure ROS2 is sourced"

# Test both middlewares
for middleware in zenoh cyclonedds; do
    echo ""
    echo -e "${BLUE}Testing $middleware middleware...${NC}"
    echo ""
    
    if [ "$middleware" = "zenoh" ]; then
        source ../setup/setup_zenoh.sh
    else
        source ../setup/setup_default.sh
    fi
    
    echo -e "${YELLOW}Start camera publisher in another terminal:${NC}"
    ROS_NODES_DIR="$(dirname "$SCRIPT_DIR")"
    echo "  cd $ROS_NODES_DIR"
    echo "  source /opt/ros/jazzy/setup.bash"
    if [ "$middleware" = "zenoh" ]; then
        echo "  source setup/setup_zenoh.sh"
    else
        echo "  source setup/setup_default.sh"
    fi
    echo "  python3 camera_publisher.py --source $VIDEO_SOURCE --fps 30"
    echo ""
    read -p "Press Enter when camera_publisher.py is running..."
    
    output_file="../output/benchmarking/${middleware}_middleware.json"
    mkdir -p "$(dirname "$output_file")"
    
    echo "Running benchmark for ${DURATION}s..."
    python3 benchmark_camera_streaming.py \
        --topic /camera/image_raw \
        --duration "$DURATION" \
        --expected-fps 30 \
        --output "$output_file" \
        --append
    
    echo ""
    echo "Stop camera_publisher.py (Ctrl+C) before next test"
    echo "Waiting 3 seconds..."
    sleep 3
done

# Compare results
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Comparison${NC}"
echo -e "${BLUE}========================================${NC}"

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

zenoh_file = Path("../output/benchmarking/zenoh_middleware.json")
cyclonedds_file = Path("../output/benchmarking/cyclonedds_middleware.json")

def load_latest(file_path):
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data[-1] if isinstance(data, list) else data
    except:
        return None

zenoh = load_latest(zenoh_file)
cyclonedds = load_latest(cyclonedds_file)

if not zenoh or not cyclonedds:
    print("ERROR: Missing benchmark results")
    print(f"  Zenoh: {zenoh_file.exists()}")
    print(f"  CycloneDDS: {cyclonedds_file.exists()}")
    exit(1)

print("\n" + "="*90)
print("MIDDLEWARE COMPARISON")
print("="*90)
print(f"\n{'Metric':<25} {'Zenoh':<20} {'CycloneDDS':<20} {'Difference':<15}")
print("-"*90)

metrics = [
    ('FPS', 'fps', '{:.2f}'),
    ('Bandwidth (Mbps)', 'bandwidth_mbps', '{:.2f}'),
    ('Avg Latency (ms)', 'avg_latency_ms', '{:.2f}'),
    ('P95 Latency (ms)', 'p95_latency_ms', '{:.2f}'),
    ('Dropped Frames', 'dropped_frames', '{:.0f}'),
]

for label, key, fmt in metrics:
    z_val = zenoh.get(key, 0)
    c_val = cyclonedds.get(key, 0)
    diff = c_val - z_val
    diff_pct = (diff / z_val * 100) if z_val != 0 else 0
    
    print(f"{label:<25} {fmt.format(z_val):<20} {fmt.format(c_val):<20} {fmt.format(diff):+.2f} ({diff_pct:+.1f}%)")

print("="*90)

# Save comparison
comparison = {
    'zenoh': zenoh,
    'cyclonedds': cyclonedds,
    'comparison': {
        'bandwidth_diff_mbps': cyclonedds.get('bandwidth_mbps', 0) - zenoh.get('bandwidth_mbps', 0),
        'bandwidth_diff_percent': ((cyclonedds.get('bandwidth_mbps', 0) - zenoh.get('bandwidth_mbps', 0)) / zenoh.get('bandwidth_mbps', 1) * 100) if zenoh.get('bandwidth_mbps', 0) > 0 else 0,
        'latency_diff_ms': cyclonedds.get('avg_latency_ms', 0) - zenoh.get('avg_latency_ms', 0),
        'latency_diff_percent': ((cyclonedds.get('avg_latency_ms', 0) - zenoh.get('avg_latency_ms', 0)) / zenoh.get('avg_latency_ms', 1) * 100) if zenoh.get('avg_latency_ms', 0) > 0 else 0,
    }
}

comparison_file = Path("../output/benchmarking/middleware_comparison.json")
comparison_file.parent.mkdir(parents=True, exist_ok=True)

with open(comparison_file, 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nComparison saved to: {comparison_file.absolute()}")
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}Benchmark comparison complete!${NC}"
