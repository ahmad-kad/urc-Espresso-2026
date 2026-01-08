# ROS Nodes Directory Structure

## Overview

The `ros_nodes/` directory is organized into clear subdirectories for better maintainability and scalability.

## Directory Structure

```
ros_nodes/
├── README.md                    # Main documentation
├── BENCHMARK_GUIDE.md          # Benchmarking guide
├── ZENOH_VS_CYCLONEDDS.md      # Middleware analysis
├── INSTALL_ZENOH.md            # Zenoh installation guide
│
├── camera_detector_node.py      # Main detector node (2106 lines)
├── camera_publisher.py          # Camera publisher with compression
├── realsense_bridge.py          # RealSense bridge
├── camera_detector.launch.py   # ROS2 launch file
├── test_setup.py               # Setup verification script
│
├── benchmarks/                  # Benchmarking tools
│   ├── __init__.py
│   ├── benchmark_compression.sh        # Unified compression benchmark
│   ├── benchmark_camera_streaming.py   # Core streaming benchmark
│   ├── benchmark_data_packets.py      # Packet size benchmark
│   ├── data_packet_publisher.py       # Packet publisher
│   ├── run_middleware_benchmark.sh    # Middleware comparison
│   └── view_benchmark_history.py      # Results viewer
│
├── setup/                       # Setup scripts
│   ├── __init__.py
│   ├── setup_default.sh        # CycloneDDS middleware setup
│   └── setup_zenoh.sh          # Zenoh middleware setup
│
├── components/                  # Modular detection components
│   ├── __init__.py
│   ├── ml_inference.py
│   ├── aruco_detector.py
│   ├── detection_merger.py
│   ├── temporal_smoother.py
│   ├── motion_tracker.py
│   ├── visualization.py
│   └── image_preprocessor.py
│
└── output/                      # Benchmark results
    └── benchmarking/
        ├── compression/
        ├── compression_cyclonedds/
        └── packet_sizes/
```

## Core Components

### Main Nodes
- **`camera_detector_node.py`**: Main ROS2 detector node with component-based architecture
- **`camera_publisher.py`**: Camera publisher supporting webcam, RealSense, video files, and compression
- **`realsense_bridge.py`**: Bridge for RealSense cameras to ROS2 topics

### Testing
- **`test_setup.py`**: Comprehensive setup verification script

## Benchmarking Tools

All benchmarking tools are in the `benchmarks/` subdirectory:

### Compression Benchmarking
- **`benchmark_compression.sh`**: Unified compression benchmark (replaces old separate scripts)
  - Tests multiple compression configurations
  - Supports both Zenoh and CycloneDDS middleware
  - Focuses on <1MB frame sizes (minimal latency impact)

### Middleware Comparison
- **`run_middleware_benchmark.sh`**: Compare Zenoh vs CycloneDDS performance

### Core Benchmarking
- **`benchmark_camera_streaming.py`**: Core camera streaming benchmark tool
- **`benchmark_data_packets.py`**: Data packet performance benchmark
- **`data_packet_publisher.py`**: Publish exact-size packets for testing
- **`view_benchmark_history.py`**: View and analyze benchmark results

## Setup Scripts

Middleware configuration scripts are in the `setup/` subdirectory:

- **`setup_default.sh`**: Configure CycloneDDS (default) middleware
- **`setup_zenoh.sh`**: Configure Zenoh middleware

## Usage Examples

### Running Benchmarks

```bash
# Compression benchmark
cd ros_nodes/benchmarks
source /opt/ros/jazzy/setup.bash
./benchmark_compression.sh 0 30 zenoh

# Middleware comparison
./run_middleware_benchmark.sh 0 30

# Packet size test
source ../setup/setup_zenoh.sh
python3 data_packet_publisher.py --packet-size-kb 512 --fps 30
```

### Using Setup Scripts

```bash
# Use CycloneDDS (default)
source ros_nodes/setup/setup_default.sh

# Use Zenoh
source ros_nodes/setup/setup_zenoh.sh
```

## Benefits of This Structure

1. **Clear Organization**: Related tools grouped together
2. **Scalability**: Easy to add new benchmarks or setup scripts
3. **Maintainability**: Single location for each type of tool
4. **Clean Root**: Core nodes remain in root, utilities in subdirectories

## Migration Notes

### Old Paths → New Paths

**Benchmark Scripts:**
- `benchmark_compression.sh` → `benchmarks/benchmark_compression.sh`
- `run_middleware_benchmark.sh` → `benchmarks/run_middleware_benchmark.sh`
- `benchmark_camera_streaming.py` → `benchmarks/benchmark_camera_streaming.py`

**Setup Scripts:**
- `setup_default.sh` → `setup/setup_default.sh`
- `setup_zenoh.sh` → `setup/setup_zenoh.sh`

**Results:**
- Results still in `output/benchmarking/` (unchanged)
- Scripts now reference paths relative to their location

