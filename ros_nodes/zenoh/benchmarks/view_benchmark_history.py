#!/usr/bin/env python3
"""
View benchmark results history
Shows all appended benchmark results with statistics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load results from file, handling both single dict and list formats"""
    if not file_path.exists():
        return []

    try:
        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []


def print_results_summary(results: List[Dict[str, Any]], title: str):
    """Print summary statistics for a list of results"""
    if not results:
        print(f"\n{title}: No results found")
        return

    print(f"\n{'='*70}")
    print(f"{title} - {len(results)} result(s)")
    print(f"{'='*70}")

    # Calculate statistics
    fps_values = [r.get("fps", 0) for r in results]
    latency_values = [r.get("avg_latency_ms", 0) for r in results]
    p95_latency_values = [r.get("p95_latency_ms", 0) for r in results]

    if fps_values:
        print(f"\nFPS Statistics:")
        print(f"  Average: {sum(fps_values)/len(fps_values):.2f}")
        print(f"  Min: {min(fps_values):.2f}")
        print(f"  Max: {max(fps_values):.2f}")

    if latency_values:
        print(f"\nAverage Latency Statistics:")
        print(f"  Average: {sum(latency_values)/len(latency_values):.2f} ms")
        print(f"  Min: {min(latency_values):.2f} ms")
        print(f"  Max: {max(latency_values):.2f} ms")

    if p95_latency_values:
        print(f"\nP95 Latency Statistics:")
        print(f"  Average: {sum(p95_latency_values)/len(p95_latency_values):.2f} ms")
        print(f"  Min: {min(p95_latency_values):.2f} ms")
        print(f"  Max: {max(p95_latency_values):.2f} ms")

    # Show individual results
    print(f"\nIndividual Results:")
    for i, result in enumerate(results, 1):
        timestamp = result.get("timestamp", "Unknown")
        rmw = result.get("rmw_implementation", "unknown")
        print(f"\n  [{i}] {timestamp} ({rmw})")
        print(f"      FPS: {result.get('fps', 0):.2f}")
        print(f"      Avg Latency: {result.get('avg_latency_ms', 0):.2f} ms")
        print(f"      P95 Latency: {result.get('p95_latency_ms', 0):.2f} ms")
        print(f"      Bandwidth: {result.get('bandwidth_mbps', 0):.2f} Mbps")
        print(f"      Dropped Frames: {result.get('dropped_frames', 0)}")


def main():
    parser = argparse.ArgumentParser(description="View benchmark results history")
    parser.add_argument(
        "--file",
        type=str,
        default="../output/benchmarking/results/benchmark_results.json",
        help="Results file to view (default: ../output/benchmarking/results/benchmark_results.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../output/benchmarking/results",
        help="Results directory (default: ../output/benchmarking/results)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all result files in results directory"
    )

    args = parser.parse_args()

    if args.all:
        # Show all results in directory
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            sys.exit(1)

        default_file = results_dir / "default_middleware.json"
        zenoh_file = results_dir / "zenoh_middleware.json"
        comparison_file = results_dir / "comparison.json"

        if default_file.exists():
            default_results = load_results(default_file)
            print_results_summary(default_results, "Default Middleware Results")

        if zenoh_file.exists():
            zenoh_results = load_results(zenoh_file)
            print_results_summary(zenoh_results, "Zenoh Middleware Results")

        if comparison_file.exists():
            comparisons = load_results(comparison_file)
            if comparisons:
                print(f"\n{'='*70}")
                print(f"Comparison History - {len(comparisons)} comparison(s)")
                print(f"{'='*70}")
                for i, comp in enumerate(comparisons, 1):
                    timestamp = comp.get("timestamp", "Unknown")
                    print(f"\n[{i}] {timestamp}")
                    if "comparison" in comp:
                        c = comp["comparison"]
                        print(
                            f"    FPS Improvement: {c.get('fps_improvement_percent', 0):.1f}%"
                        )
                        print(
                            f"    Latency Reduction: {c.get('latency_reduction_percent', 0):.1f}%"
                        )
    else:
        # Show single file
        file_path = Path(args.file)
        results = load_results(file_path)

        if not results:
            print(f"No results found in {file_path}")
            sys.exit(1)

        print_results_summary(results, f"Results from {file_path.name}")


if __name__ == "__main__":
    main()
