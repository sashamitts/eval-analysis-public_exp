#!/usr/bin/env python3
"""
Setup verification script for NAME MVP.

Checks that all required dependencies are installed and provides
helpful instructions if anything is missing.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check that all required packages are available."""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'yaml': 'pyyaml',
    }

    missing = []

    print("Checking dependencies...")
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (missing)")
            missing.append(package_name)

    if missing:
        print("\n❌ Missing dependencies!")
        print("\nTo install, run:")
        print("  poetry install")
        print("\nOr if using pip:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("\n✅ All dependencies installed!")
    return True

def check_data_files():
    """Check that NAME data files exist."""
    required_files = [
        'data/external/name_tasks.yaml',
        'data/external/name_runs.jsonl',
        'data/wrangled/name_learning_curves.csv',
        'data/wrangled/name_horizons.csv',
    ]

    print("\nChecking data files...")
    missing = []

    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {filepath} ({size:.1f} MB)")
        else:
            print(f"  ✗ {filepath} (missing)")
            missing.append(filepath)

    if missing:
        print("\n⚠️  Missing data files!")
        print("\nTo generate, run:")
        print("  poetry run dvc repro plot_name_results")
        print("\nOr run individual stages:")
        print("  poetry run python -m src.generate.synthetic_tasks")
        print("  poetry run python -m src.generate.generate_runs")
        print("  poetry run python -m src.wrangle.name_metrics")
        print("  poetry run python -m src.plot.name_plots")
        return False

    print("\n✅ All data files present!")
    return True

def check_plots():
    """Check that visualization plots exist."""
    plot_files = [
        'plots/name_learning_curves_sample.png',
        'plots/name_ai_vs_human_first_attempt.png',
        'plots/name_n_attempt_horizons.png',
        'plots/name_difficulty_distribution.png',
    ]

    print("\nChecking plots...")
    missing = []

    for filepath in plot_files:
        path = Path(filepath)
        if path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (missing)")
            missing.append(filepath)

    if missing:
        print("\n⚠️  Missing plots!")
        print("\nTo generate, run:")
        print("  poetry run python -m src.plot.name_plots")
        return False

    print("\n✅ All plots generated!")
    return True

def main():
    """Run all setup checks."""
    print("=" * 70)
    print("NAME MVP Setup Verification")
    print("=" * 70)

    deps_ok = check_dependencies()

    if not deps_ok:
        print("\n" + "=" * 70)
        print("Setup incomplete. Please install dependencies first.")
        print("=" * 70)
        sys.exit(1)

    data_ok = check_data_files()
    plots_ok = check_plots()

    print("\n" + "=" * 70)

    if deps_ok and data_ok and plots_ok:
        print("✅ SETUP COMPLETE!")
        print("\nYou can now:")
        print("  • Run exploration: poetry run python explore_name_results.py")
        print("  • Open notebook: poetry run jupyter notebook explore_name_interactive.ipynb")
        print("  • View plots in: plots/name_*.png")
        print("  • Re-run pipeline: poetry run dvc repro plot_name_results")
    elif deps_ok and data_ok:
        print("⚠️  SETUP MOSTLY COMPLETE")
        print("\nPlots are missing but data exists. Run:")
        print("  poetry run python -m src.plot.name_plots")
    elif deps_ok:
        print("⚠️  DEPENDENCIES OK, DATA MISSING")
        print("\nGenerate data by running:")
        print("  poetry run dvc repro plot_name_results")

    print("=" * 70)

if __name__ == "__main__":
    main()
