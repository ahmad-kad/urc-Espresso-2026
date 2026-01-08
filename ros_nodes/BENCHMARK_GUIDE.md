# ROS2 Benchmarking Guide

## Overview

This guide covers all benchmarking tools for ROS2 camera streaming and data packet performance. The infrastructure has been streamlined based on findings that **frame sizes <1MB have minimal latency impact**.

## Quick Start

### 1. Compression Benchmark (Simplified)

Test compression levels with any middleware:

```bash
cd ros_nodes/benchmarks
source /opt/ros/jazzy/setup.bash

# Test with Zenoh
./benchmark_compression.sh 0 30 zenoh

# Test with CycloneDDS
./benchmark_compression.sh 0 30 cyclonedds
```

**Key Finding**: Configurations producing <1MB frames show minimal latency impact, so the suite focuses on practical compression levels.

### 2. Middleware Comparison

Compare Zenoh vs CycloneDDS:

```bash
cd ros_nodes/benchmarks
./run_middleware_benchmark.sh 0 30
```

### 3. Packet Size Benchmark

Test specific packet sizes (128KB-2048KB):

```bash
cd ros_nodes
source /opt/ros/jazzy/setup.bash
source setup/setup_zenoh.sh  # or setup/setup_default.sh

# Terminal 1: Publisher
cd benchmarks
python3 data_packet_publisher.py --packet-size-kb 512 --fps 30

# Terminal 2: Benchmark
python3 benchmark_data_packets.py --duration 30 --output ../output/benchmarking/packet_sizes/results/test_512kb.json
```

## Benchmark Tools

### `benchmark_compression.sh` (Unified)

**Purpose**: Test compression configurations with configurable middleware.

**Location**: `benchmarks/benchmark_compression.sh`

**Usage**:
```bash
cd ros_nodes/benchmarks
./benchmark_compression.sh [VIDEO_SOURCE] [DURATION] [MIDDLEWARE]
```

**Parameters**:
- `VIDEO_SOURCE`: Camera index (e.g., `0`) or video file path (default: `0`)
- `DURATION`: Test duration in seconds (default: `30`)
- `MIDDLEWARE`: `zenoh` or `cyclonedds` (default: `zenoh`)

**Test Configurations**:
- `raw_full`: Raw 640x480 (baseline)
- `jpeg85_full`: JPEG Q85, 640x480 (~18-28 Mbps)
- `jpeg75_full`: JPEG Q75, 640x480 (~12-18 Mbps)
- `jpeg85_half`: JPEG Q85, 320x240 (~4-6 Mbps)
- `jpeg75_quarter`: JPEG Q75, 160x120 (~0.5-1 Mbps)

**Results**: Saved to `output/benchmarking/compression_{middleware}/results/`

### `run_middleware_benchmark.sh`

**Purpose**: Compare Zenoh vs CycloneDDS middleware performance.

**Location**: `benchmarks/run_middleware_benchmark.sh`

**Usage**:
```bash
cd ros_nodes/benchmarks
./run_middleware_benchmark.sh [VIDEO_SOURCE] [DURATION]
```

**Results**: 
- Individual: `output/benchmarking/{middleware}_middleware.json`
- Comparison: `output/benchmarking/middleware_comparison.json`

**Location**: `benchmarks/benchmark_camera_streaming.py`

### `benchmark_camera_streaming.py`

**Purpose**: Core benchmarking tool for camera streams.

**Usage**:
```bash
python3 benchmark_camera_streaming.py \
    --topic /camera/image_raw \
    --duration 30 \
    --expected-fps 30 \
    --output results.json \
    --append
```

**Options**:
- `--topic`: ROS topic to subscribe to
- `--duration`: Benchmark duration in seconds
- `--expected-fps`: Expected frames per second
- `--output`: Output JSON file path
- `--append`: Append results to existing file

### `benchmark_data_packets.py`

**Purpose**: Benchmark data packet performance (for packet size testing).

**Usage**:
```bash
python3 benchmark_data_packets.py \
    --topic /data_packets \
    --duration 30 \
    --expected-fps 30 \
    --output results.json \
    --append
```

**Location**: `benchmarks/benchmark_data_packets.py`

**Options**: Same as `benchmark_camera_streaming.py`

### `data_packet_publisher.py`

**Purpose**: Publish exact-size data packets for network testing.

**Location**: `benchmarks/data_packet_publisher.py`

**Usage**:
```bash
cd ros_nodes/benchmarks
python3 data_packet_publisher.py --packet-size-kb 512 --fps 30
```

**Valid packet sizes**: 128, 256, 512, 1024, 2048 KB

## Understanding Results

### Key Metrics

- **FPS**: Frames/packets per second received
- **Bandwidth**: Mbps usage
- **Avg Latency**: Average inter-frame/packet interval (ms)
- **P95 Latency**: 95th percentile latency (ms)
- **Dropped Frames/Packets**: Loss count
- **Frame/Packet Size**: Average size in KB

### Latency Impact Finding

**Key Finding**: Frame sizes <1MB show minimal latency impact.

This means:
- Compression that keeps frames <1MB is recommended
- Resolution reduction (half, quarter) is effective
- JPEG quality 75-85 provides good balance
- Further compression below ~1MB has diminishing returns for latency

### Compression Recommendations

Based on benchmark results:

1. **High Quality** (if bandwidth allows): `jpeg85_full` (~18-28 Mbps)
2. **Balanced**: `jpeg75_full` (~12-18 Mbps)
3. **Low Bandwidth**: `jpeg85_half` (~4-6 Mbps) or `jpeg75_quarter` (~0.5-1 Mbps)

All recommended configurations keep frames <1MB, minimizing latency impact.

## Results Structure

### Compression Benchmarks

```
ros_nodes/output/benchmarking/compression_{middleware}/
  results/
    raw_full.json
    jpeg85_full.json
    jpeg75_full.json
    jpeg85_half.json
    jpeg75_quarter.json
    compression_summary.json
```

### Middleware Comparison

```
ros_nodes/output/benchmarking/
  zenoh_middleware.json
  cyclonedds_middleware.json
  middleware_comparison.json
```

### Packet Size Benchmarks

```
ros_nodes/output/benchmarking/packet_sizes/
  results/
    {middleware}_{size}kb.json
```

## Best Practices

1. **Use unified scripts**: Prefer `benchmark_compression.sh` over old separate scripts
2. **Test duration**: 30 seconds is usually sufficient for stable metrics
3. **Middleware consistency**: Always use the same middleware for publisher and subscriber
4. **Frame size awareness**: Target <1MB frames for minimal latency impact
5. **Append results**: Use `--append` flag to build result history

## Troubleshooting

### No packets/frames received
- Ensure publisher and subscriber use **same middleware**
- Check topic names match
- Verify ROS2 discovery: `ros2 topic list`

### Unexpected results
- Check middleware: `echo $RMW_IMPLEMENTATION`
- Verify camera/video source is working
- Check network conditions for packet size tests

### Missing dependencies
```bash
pip install rclpy sensor-msgs cv-bridge std-msgs vision-msgs
```

## Middleware Selection

For video streaming with compression, see [ZENOH_VS_CYCLONEDDS.md](ZENOH_VS_CYCLONEDDS.md) for detailed analysis.

**Quick recommendation**: Use CycloneDDS (default) for compressed video streaming. Zenoh is only worth it for raw/uncompressed streaming or complex network topologies.

