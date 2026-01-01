#!/usr/bin/env python3
"""
Development environment setup script
Installs all development dependencies and sets up pre-commit hooks
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status"""
    print(f"{description}...")
    try:
        subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main setup function"""
    print("Setting up YOLO AI Camera development environment")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)

    success_count = 0
    total_steps = 5

    # Step 1: Install development dependencies
    if run_command(
        f"{sys.executable} -m pip install -e .",
        "Installing project in development mode",
    ):
        success_count += 1

    # Step 2: Install additional dev tools
    dev_packages = [
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ]

    if run_command(
        f"{sys.executable} -m pip install {' '.join(dev_packages)}",
        "Installing code quality tools",
    ):
        success_count += 1

    # Step 3: Install pre-commit hooks
    if run_command("pre-commit install", "Installing pre-commit hooks"):
        success_count += 1

    # Step 4: Run initial code formatting
    if run_command("pre-commit run --all-files", "Running initial code formatting"):
        success_count += 1

    # Step 5: Verify installation
    if run_command(
        f"{sys.executable} -c \"import utils; print('Utils import successful')\"",
        "Verifying installation",
    ):
        success_count += 1

    print("\n" + "=" * 60)
    print(f"Setup completed: {success_count}/{total_steps} steps successful")

    if success_count == total_steps:
        print("\nDevelopment environment is ready!")
        print("\nNext steps:")
        print("  • Run 'python scripts/benchmark_models.py' to test the setup")
        print("  • Use 'pre-commit run --all-files' before committing")
        print("  • Run 'mypy .' to check types")
        print("  • Run 'flake8 .' to check code style")
    else:
        print(f"\nSetup completed with {total_steps - success_count} issues")
        print("   Please check the error messages above and fix any issues")

    return success_count == total_steps


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
