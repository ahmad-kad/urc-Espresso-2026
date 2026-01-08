# Installing Zenoh Middleware for ROS2

Zenoh is an optional middleware for ROS2 that can provide better performance in some scenarios. The benchmark suite will work fine without it, but if you want to compare Zenoh vs Default middleware, you'll need to install it.

## Quick Check

To see if Zenoh is already installed:
```bash
ros2 doctor --report | grep -i zenoh
```

Or try:
```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
python3 -c "import rclpy; rclpy.init(); print('Zenoh available')"
```

## Installation for ROS2 Jazzy

### Option 1: Pre-built Package (RECOMMENDED - Easiest)

Install the pre-built Zenoh packages (you need both):
```bash
sudo apt update
sudo apt install -y ros-jazzy-zenoh-cpp-vendor ros-jazzy-rmw-zenoh-cpp
```

- `ros-jazzy-zenoh-cpp-vendor`: Provides the underlying Zenoh C++ library
- `ros-jazzy-rmw-zenoh-cpp`: Provides the ROS2 middleware plugin (`librmw_zenoh_cpp.so`)

**Both packages are required!** The vendor package alone is not sufficient.

**Verify installation:**
```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
python3 -c "import rclpy; rclpy.init(); print('✓ Zenoh installed successfully')"
```

If you see any security tools or want additional features:
```bash
sudo apt install -y ros-jazzy-zenoh-security-tools
```

### Option 2: Build from Source (if package doesn't work)

1. **Install Dependencies:**
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libasio-dev \
    libtinyxml2-dev \
    pkg-config
```

2. **Install Rust (required for Zenoh):**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

3. **Clone and Build Zenoh Plugin:**
```bash
cd ~
git clone https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds.git
cd zenoh-plugin-ros2dds
mkdir build && cd build
cmake ..
make
sudo make install
```

4. **Verify Installation:**
```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
python3 -c "import rclpy; rclpy.init(); print('✓ Zenoh installed successfully')"
```

## Using Zenoh

Once installed, you can use Zenoh in your benchmarks:

```bash
source /opt/ros/jazzy/setup.bash
source ros_nodes/setup_zenoh.sh
python3 ros_nodes/benchmark_camera_streaming.py
```

Or use the automated benchmark suite:
```bash
./ros_nodes/run_middleware_benchmark.sh
```

## Troubleshooting

**"librmw_zenoh_cpp.so: cannot open shared object file"**
- Zenoh plugin is not installed or not in library path
- Try: `sudo ldconfig` after installation
- Check: `ldconfig -p | grep zenoh`

**"RMW implementation not found"**
- Make sure you sourced ROS2: `source /opt/ros/jazzy/setup.bash`
- Verify installation: `ros2 doctor --report`

**Build errors:**
- Ensure all dependencies are installed
- Check Rust version: `rustc --version` (should be recent)
- Try cleaning build: `rm -rf build && mkdir build`

## Note

Zenoh is **optional**. The benchmark suite will work perfectly fine with just the default middleware (FastRTPS/CycloneDDS). Zenoh is useful for:
- Cross-network communication
- Lower latency in some scenarios
- Better performance with many nodes
- Comparison testing

For most use cases, the default middleware is sufficient.

