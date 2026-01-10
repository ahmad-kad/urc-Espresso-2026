#!/usr/bin/env python3
"""
Dual Camera Launch Script for LiDAR-Camera Fusion
Launches two camera nodes with maximum quality configurations
"""

import subprocess
import sys
import time
import signal
import os

# ============================================================================
# CONFIGURATION - All quality/performance settings in one place
# ============================================================================
CONFIG = {
    # Camera settings - optimized for quality over framerate
    "width": 1920,           # Full HD for detail
    "height": 1080,          # Full HD for detail
    "fps": 15.0,             # Lower framerate allows better quality/exposure
    "format": "MJPEG",       # Compressed but lossless for network efficiency
    
    # Device scanning
    "camera_scan_range": 10,  # How many /dev/videoX to check
    
    # Timing (seconds)
    "process_kill_timeout": 3,
    "ros_daemon_wait": 2,
    "camera_init_wait": 5,
    "inter_launch_delay": 3,
    "topic_check_timeout": 5,
    "hz_check_timeout": 3,
    
    # ROS settings
    "ros_distro": "jazzy",
    "working_dir": "/home/durian/glitter/glitter",
    "camera_node_script": "src/camera/pi_camera_node.py",
    
    # Topic naming
    "topic_namespace": "cameras",
    "left_topic_name": "left/image_raw",
    "right_topic_name": "right/image_raw",
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_ros_source_cmd():
    """Return source command for ROS environment"""
    return f"source /opt/ros/{CONFIG['ros_distro']}/setup.bash"

def run_command(cmd, name):
    """Run a command in the background with full shell environment"""
    print(f"Starting {name}...")
    print(f"Command: {cmd}")
    full_cmd = f"bash -c '{cmd}'"
    process = subprocess.Popen(
        full_cmd,
        shell=True,
        preexec_fn=os.setsid
    )
    return process

def run_command_sync(cmd, timeout=None, capture=True):
    """Run a command synchronously and return output"""
    try:
        result = subprocess.run(
            f"bash -c '{cmd}'",
            shell=True,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"Command execution error: {e}")
        return None

def kill_process_group(pid, timeout=None):
    """Safely kill a process group"""
    if timeout is None:
        timeout = CONFIG["process_kill_timeout"]
    
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        process = subprocess.Popen(
            ["sleep", str(timeout)],
            preexec_fn=os.setsid
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass

def clean_existing_camera_processes():
    """Kill any existing camera processes to free up devices"""
    patterns = ["pi_camera_node", "camera_left", "camera_right"]
    running_pids = set()
    
    for pattern in patterns:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        running_pids.update(pids)
    
    if running_pids:
        print(f"Cleaning {len(running_pids)} existing camera processes...")
        for pid in running_pids:
            kill_process_group(int(pid))
        time.sleep(CONFIG["process_kill_timeout"])

# ============================================================================
# CAMERA DETECTION
# ============================================================================

def detect_cameras():
    """
    Detect available cameras via v4l2 device enumeration
    Returns list of working camera indices
    """
    # Use v4l2-ctl which works better with industrial cameras
    result = run_command_sync("v4l2-ctl --list-devices", timeout=5)
    
    if not result or not result.stdout.strip():
        print("v4l2-ctl not available, falling back to device scan")
        return fallback_camera_detection()
    
    # Parse v4l2-ctl output to find /dev/videoX devices
    indices = []
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line.startswith('/dev/video'):
            try:
                # Extract number from /dev/videoN
                idx = int(line.split('video')[1].split()[0])
                if idx not in indices and idx % 2 == 0:  # Use only even indices (main streams)
                    indices.append(idx)
            except (ValueError, IndexError):
                continue
    
    return sorted(indices)

def fallback_camera_detection():
    """
    Fallback detection using cv2.VideoCapture for when v4l2-ctl unavailable
    """
    camera_detection_code = f"""
import cv2
cameras = []

for i in range({CONFIG['camera_scan_range']}):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        continue
    
    # Try to read a frame - just need one successful frame
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        h, w = frame.shape[:2]
        cameras.append((i, w, h))
    
    cap.release()

if cameras:
    for idx, w, h in cameras:
        print(f"Camera {{idx}}: {{w}}x{{h}}")
else:
    print("No cameras found")
"""
    
    result = run_command_sync(camera_detection_code, timeout=30)
    if not result or not result.stdout.strip() or "No cameras found" in result.stdout:
        return []
    
    indices = []
    for line in result.stdout.strip().split('\n'):
        try:
            idx = int(line.split(':')[0].split(' ')[1])
            indices.append(idx)
        except (ValueError, IndexError):
            continue
    
    return indices

def validate_camera_count(available_indices):
    """Warn if insufficient cameras, but allow single camera mode"""
    count = len(available_indices)
    
    if count == 0:
        print("✗ No cameras detected")
        return False
    
    print(f"\n✓ Found {count} working camera(s):")
    for idx in available_indices:
        print(f"  - /dev/video{idx}")
    
    if count == 1:
        print("⚠ Single camera mode - fusion limited to one perspective")
    
    return True

# ============================================================================
# ROS OPERATIONS
# ============================================================================

def check_topics():
    """Check if camera topics are publishing"""
    cmd = (
        f"{get_ros_source_cmd()} && "
        f"ros2 topic list | grep -E '{CONFIG['topic_namespace']}/(left|right)'"
    )
    result = run_command_sync(cmd, timeout=CONFIG["topic_check_timeout"])
    
    if not result or not result.stdout.strip():
        return []
    
    topics = [t.strip() for t in result.stdout.strip().split('\n') if t.strip()]
    return topics

def get_topic_hz(topic):
    """Get publishing rate of a topic"""
    cmd = (
        f"{get_ros_source_cmd()} && "
        f"timeout {CONFIG['hz_check_timeout']} ros2 topic hz {topic}"
    )
    result = run_command_sync(cmd, timeout=CONFIG["hz_check_timeout"] + 1)
    
    if not result or not result.stdout.strip():
        return None
    
    for line in result.stdout.split('\n'):
        if 'average rate:' in line:
            return line.split('average rate:')[1].strip()
    
    return None

def start_ros_daemon():
    """Start ROS 2 daemon"""
    cmd = f"{get_ros_source_cmd()} && ros2 daemon start"
    subprocess.run(f"bash -c '{cmd}'", shell=True, capture_output=True)
    time.sleep(CONFIG["ros_daemon_wait"])

# ============================================================================
# CAMERA NODE LAUNCHING
# ============================================================================

def build_camera_command(device_id, side):
    """
    Build camera node launch command
    Args:
        device_id: /dev/videoX index
        side: "left" or "right"
    """
    topic_name = (
        CONFIG["left_topic_name"] if side == "left"
        else CONFIG["right_topic_name"]
    )
    
    cmd = (
        f"cd {CONFIG['working_dir']} && "
        f"{get_ros_source_cmd()} && "
        f"python3 {CONFIG['camera_node_script']} "
        f"--ros-args "
        f"-p device_id:={device_id} "
        f"-p topic_name:={CONFIG['topic_namespace']}/{topic_name} "
        f"-p width:={CONFIG['width']} "
        f"-p height:={CONFIG['height']} "
        f"-p fps:={CONFIG['fps']} "
        f"-p format:={CONFIG['format']} "
        f"--remap __node:=camera_{side}_node"
    )
    
    return cmd

def launch_cameras(available_indices):
    """Launch camera nodes and return process list"""
    processes = []
    
    print("\n" + "="*60)
    print("Camera Configuration (Quality-Optimized):")
    print(f"  Resolution: {CONFIG['width']}x{CONFIG['height']}")
    print(f"  Framerate: {CONFIG['fps']} FPS")
    print(f"  Format: {CONFIG['format']}")
    print("="*60 + "\n")
    
    if len(available_indices) >= 1:
        idx = available_indices[0]
        cmd = build_camera_command(idx, "left")
        processes.append(run_command(cmd, f"Camera Left (Device {idx})"))
        time.sleep(CONFIG["inter_launch_delay"])
    
    if len(available_indices) >= 2:
        idx = available_indices[1]
        cmd = build_camera_command(idx, "right")
        processes.append(run_command(cmd, f"Camera Right (Device {idx})"))
        time.sleep(CONFIG["inter_launch_delay"])
    
    return processes

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("Dual Camera Launch for LiDAR Fusion (Quality Mode)")
    print("="*60)
    
    # Clean up existing processes
    clean_existing_camera_processes()
    
    # Detect cameras
    print("\nDetecting cameras...")
    available_indices = detect_cameras()
    
    if not validate_camera_count(available_indices):
        print("Cannot proceed without cameras")
        return
    
    # Start ROS
    print("\nStarting ROS 2 daemon...")
    start_ros_daemon()
    
    # Launch camera nodes
    processes = []
    
    try:
        processes = launch_cameras(available_indices)
        
        # Wait for initialization
        print(f"Waiting {CONFIG['camera_init_wait']}s for cameras to initialize...")
        time.sleep(CONFIG["camera_init_wait"])
        
        # Check topics
        topics = check_topics()
        
        if topics:
            print(f"\n✓ Camera topics publishing ({len(topics)}):")
            for topic in topics:
                print(f"  - {topic}")
                hz = get_topic_hz(topic)
                if hz:
                    print(f"    Publishing at: {hz} Hz")
        else:
            print("\n✗ No camera topics found after initialization")
        
        # Summary
        print("\n" + "="*60)
        print("Dual Camera System Running!")
        print("="*60)
        if len(available_indices) >= 1:
            print(f"  - {CONFIG['topic_namespace']}/{CONFIG['left_topic_name']}")
        if len(available_indices) >= 2:
            print(f"  - {CONFIG['topic_namespace']}/{CONFIG['right_topic_name']}")
        
        print("\nFusion Examples:")
        print(f"  python3 src/core/fusion.py --ros-args -p image_topic:={CONFIG['topic_namespace']}/{CONFIG['left_topic_name']}")
        print(f"  python3 src/core/fusion.py --ros-args -p image_topic:={CONFIG['topic_namespace']}/{CONFIG['right_topic_name']}")
        
        print("\nPress Ctrl+C to stop...")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Stopping camera processes...")
        for process in processes:
            kill_process_group(process.pid)
        print("All cameras stopped.")

if __name__ == "__main__":
    main()