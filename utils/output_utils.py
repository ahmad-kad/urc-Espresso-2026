"""
Utilities for organizing and formatting script outputs
Ensures consistent output structure for debugging, testing, and sanity checking
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages organized output directories and file saving"""

    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.subdirs = {
            "benchmarking": self.base_dir / "benchmarking",
            "evaluation": self.base_dir / "evaluation",
            "debug": self.base_dir / "debug",
            "testing": self.base_dir / "testing",
            "models": self.base_dir / "models",
            "visualization": self.base_dir / "visualization",
        }

        # Create all directories
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self, data: Dict[str, Any], filename: str, category: str = "debug"
    ) -> Path:
        """Save data as JSON to organized directory"""
        if category not in self.subdirs:
            category = "debug"

        output_path = self.subdirs[category] / f"{filename}.json"

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def save_csv(
        self, data: List[Dict[str, Any]], filename: str, category: str = "evaluation"
    ) -> Path:
        """Save data as CSV to organized directory"""
        if category not in self.subdirs:
            category = "evaluation"

        output_path = self.subdirs[category] / f"{filename}.csv"

        if data:
            fieldnames = data[0].keys()
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def save_text_report(
        self, content: str, filename: str, category: str = "debug"
    ) -> Path:
        """Save text report to organized directory"""
        if category not in self.subdirs:
            category = "debug"

        output_path = self.subdirs[category] / f"{filename}.txt"

        with open(output_path, "w") as f:
            f.write(content)

        logger.info(f"Report saved to: {output_path}")
        return output_path

    def get_path(self, category: str, filename: str) -> Path:
        """Get full path for a file in a category directory"""
        if category not in self.subdirs:
            category = "debug"
        return self.subdirs[category] / filename


# Global output manager instance
output_manager = OutputManager()


def save_benchmark_results(results: Dict[str, Any], model_name: str) -> Path:
    """Save benchmarking results with consistent formatting"""
    return output_manager.save_json(results, f"benchmark_{model_name}", "benchmarking")


def save_evaluation_results(results: Dict[str, Any], model_name: str) -> Path:
    """Save evaluation results with consistent formatting"""
    return output_manager.save_json(results, f"evaluation_{model_name}", "evaluation")


def save_debug_info(info: Dict[str, Any], script_name: str) -> Path:
    """Save debug information"""
    return output_manager.save_json(info, f"debug_{script_name}", "debug")


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format metrics as a nice text table"""
    table = f"\n{title}\n{'='*50}\n"

    for key, value in metrics.items():
        if isinstance(value, float):
            table += f"{key:<20}: {value:.4f}\n"
        else:
            table += f"{key:<20}: {value}\n"

    return table


def print_section_header(title: str, width: int = 60):
    """Print a formatted section header"""
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}")
