#!/usr/bin/env python3
"""
Check the status of the running rover optimization pipeline
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

def check_log_status():
    """Check the current status from the log file"""
    output_dir = Path("output")
    log_file = output_dir / "rover_optimization.log"

    if not os.path.exists(log_file):
        print("ERROR: No log file found. The optimization may not have started yet.")
        return

    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        if not lines:
            print("Log file exists but is empty.")
            return

        # Get the last few lines
        recent_lines = lines[-10:]

        print("CURRENT OPTIMIZATION STATUS")
        print("="*50)
        print(f"Log file: {log_file}")
        print(f"Last updated: {datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Check current phase
        current_phase = "Unknown"
        for line in reversed(lines):
            if "PHASE" in line and "INFO" in line:
                if "Multi-size training" in line:
                    current_phase = "Phase 1: Multi-size Training"
                elif "Quantization" in line:
                    current_phase = "Phase 2: Quantization"
                elif "benchmarking" in line:
                    current_phase = "Phase 3: Benchmarking"
                elif "recommendations" in line:
                    current_phase = "Phase 4: Recommendations"
                break

        print(f"Current Phase: {current_phase}")
        print()

        # Check current architecture and size
        current_arch = "Unknown"
        current_size = "Unknown"

        for line in reversed(lines):
            if "ARCHITECTURE:" in line:
                current_arch = line.split("ARCHITECTURE:")[-1].strip()
                break

        for line in reversed(lines):
            if "TRAINING SIZE:" in line:
                current_size = line.split("TRAINING SIZE:")[-1].strip()
                break

        if current_arch != "Unknown":
            print(f"Current Architecture: {current_arch}")
        if current_size != "Unknown":
            print(f"Current Size: {current_size}")
        print()

        # Show recent activity
        print("Recent Activity:")
        for line in recent_lines:
            if "INFO" in line:
                # Clean up the line
                clean_line = line.split("INFO - ")[-1].strip()
                if clean_line and len(clean_line) > 10:  # Skip very short messages
                    print(f"  • {clean_line}")

        print()

        # Check for completion
        if any("COMPLETED SUCCESSFULLY" in line for line in lines):
            print("OPTIMIZATION COMPLETED!")
            check_results()
        elif any("failed" in line.lower() for line in lines[-20:]):
            print("OPTIMIZATION FAILED - Check log for details")
        else:
            print("OPTIMIZATION IN PROGRESS...")
            print("Run this script again in a few minutes to check progress.")

    except Exception as e:
        print(f"Error reading log: {str(e)}")


def check_results():
    """Check if results files exist"""
    output_dir = Path("output")
    result_files = [
        output_dir / "rover_optimization_summary.txt",
        output_dir / "rover_benchmark_results.csv"
    ]

    print("Results Files:")
    for file in result_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  FOUND: {file} ({size} bytes)")
        else:
            print(f"  MISSING: {file}")


def check_model_outputs():
    """Check for newly created model directories"""
    output_dir = Path("output/models")

    if not output_dir.exists():
        print("No output/models directory found")
        return

    print("New Model Directories:")
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]

    # Sort by modification time (newest first)
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    for subdir in subdirs[:10]:  # Show last 10
        mtime = datetime.fromtimestamp(subdir.stat().st_mtime)
        weights_dir = subdir / "weights"
        if weights_dir.exists():
            weights = list(weights_dir.glob("*.pt"))
            print(f"  {subdir.name} ({mtime.strftime('%H:%M')}) - {len(weights)} weights")


def main():
    """Main status checker"""
    print("ROVER OPTIMIZATION STATUS CHECKER")
    print("="*50)
    print()

    check_log_status()
    print()
    check_model_outputs()
    print()
    check_results()


if __name__ == '__main__':
    main()
