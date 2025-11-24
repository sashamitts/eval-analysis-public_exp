# NAME MVP Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

Using Poetry (recommended):
```bash
poetry install
```

Or using pip:
```bash
pip install pandas numpy matplotlib scipy pyyaml scikit-learn
```

### 2. Verify Setup

```bash
poetry run python check_setup.py
```

This will check:
- ✓ All dependencies installed
- ✓ Data files generated
- ✓ Plots created

### 3. Generate Data (if needed)

If data files are missing:
```bash
poetry run dvc repro plot_name_results
```

This will:
1. Generate 100 synthetic tasks
2. Simulate 56,000 runs (50 novices + 3 AI models)
3. Fit learning curves
4. Calculate NAME horizons
5. Create visualizations

**Expected time:** ~2-3 minutes

## Exploring the Results

### Option 1: Quick Overview (Command Line)

```bash
poetry run python explore_name_results.py
```

Shows:
- Task difficulty distribution
- Learning curve statistics
- Model performance comparison
- Sample learning trajectories

### Option 2: Interactive Exploration (Jupyter)

```bash
poetry run jupyter notebook explore_name_interactive.ipynb
```

Interactive features:
- Inspect individual learning curves
- Compare AI models
- Simulate custom AI capabilities
- Analyze by domain
- Experiment with parameters

### Option 3: View Plots Directly

Open these files in your image viewer:
- `plots/name_learning_curves_sample.png`
- `plots/name_ai_vs_human_first_attempt.png`
- `plots/name_n_attempt_horizons.png`
- `plots/name_difficulty_distribution.png`

### Option 4: Examine Raw Data

```bash
# View task definitions
cat data/external/name_tasks.yaml | head -30

# View learning curve results
cat data/wrangled/name_learning_curves.csv | head -10

# View NAME horizons
cat data/wrangled/name_horizons.csv

# Count total runs
wc -l data/external/name_runs.jsonl
```

## Understanding the Output

### NAME Horizons (Main Metric)

From `data/wrangled/name_horizons.csv`:

| Model | N-Attempt Horizon (50%) | Interpretation |
|-------|------------------------|----------------|
| baseline-weak | ~1.2 attempts | Below novice first-attempt |
| baseline-medium | ~0.4 attempts | Near novice first-attempt |
| baseline-strong | ~0.3 attempts | Above novice first-attempt |

**Reading the metric:**
- **< 1.0**: AI performs better than novice first attempt
- **= 1.0**: AI matches novice first attempt
- **> 1.0**: AI performs worse than novice first attempt

### Learning Curves

From `data/wrangled/name_learning_curves.csv`:

Each row represents one task's learning curve fit:
- `first_attempt_success_rate`: Task difficulty (0-1)
- `fitted_lambda`: Learning speed (~0.2-0.6)
- `fitted_asymptote`: Maximum achievable success (~0.8-1.0)
- `n_for_50pct`: Attempts needed to reach 50% success
- `rmse`: Fit quality (good if < 0.1)

## Troubleshooting

### "Missing dependencies" error

**Solution 1 (Poetry):**
```bash
poetry install
poetry run python explore_name_results.py
```

**Solution 2 (pip):**
```bash
pip install pandas numpy matplotlib scipy pyyaml scikit-learn
python explore_name_results.py
```

### "Missing data files" error

**Generate data:**
```bash
poetry run dvc repro plot_name_results
```

Or run stages individually:
```bash
poetry run python -m src.generate.synthetic_tasks
poetry run python -m src.generate.generate_runs
poetry run python -m src.wrangle.name_metrics
poetry run python -m src.plot.name_plots
```

### "ModuleNotFoundError" in Jupyter

**Install Jupyter in Poetry environment:**
```bash
poetry add jupyter --group dev
poetry run jupyter notebook explore_name_interactive.ipynb
```

### DVC pipeline fails

**Check DVC installation:**
```bash
poetry run dvc version
```

If not installed:
```bash
poetry install  # Should already include DVC
```

## Next Steps

### Experiment with Parameters

1. **Change task difficulty:**
   - Edit `src/generate/synthetic_tasks.py`
   - Modify `difficulty_range=(0.05, 0.8)` to your preferred range
   - Re-run: `poetry run dvc repro plot_name_results`

2. **Adjust learning speed:**
   - Edit `src/generate/synthetic_tasks.py`
   - Modify `default_lambda=0.4` (higher = faster learning)
   - Re-run pipeline

3. **Add more AI models:**
   - Edit `src/generate/generate_runs.py`
   - Add to `ai_models` list with custom capability
   - Re-run pipeline

### Use Your Own Data

See `docs/NAME_README.md` for detailed instructions on:
- Adding real AI models
- Collecting real human data
- Adding new task domains
- Implementing LLM judges

## Getting Help

1. **Check setup:** `poetry run python check_setup.py`
2. **Read full docs:** `docs/NAME_README.md`
3. **Review implementation:** `docs/NAME_IMPLEMENTATION_PLAN.md`
4. **Run example:** `poetry run python explore_name_results.py`

## File Structure Reference

```
eval-analysis-public_exp/
├── check_setup.py                      # Setup verification
├── explore_name_results.py             # Quick exploration
├── explore_name_interactive.ipynb      # Interactive notebook
│
├── src/
│   ├── generate/
│   │   ├── synthetic_tasks.py          # Task generator
│   │   └── generate_runs.py            # Run generator
│   ├── simulate/
│   │   └── novice_learning.py          # Learning simulator
│   ├── wrangle/
│   │   └── name_metrics.py             # Metric calculator
│   └── plot/
│       └── name_plots.py               # Visualizations
│
├── data/
│   ├── external/
│   │   ├── name_tasks.yaml             # 100 tasks
│   │   └── name_runs.jsonl             # 56,000 runs
│   └── wrangled/
│       ├── name_learning_curves.csv    # 100 curves
│       └── name_horizons.csv           # 3 horizons
│
├── plots/
│   └── name_*.png                      # 4 visualizations
│
└── docs/
    ├── NAME_README.md                  # Full documentation
    └── NAME_IMPLEMENTATION_PLAN.md     # Architecture details
```

---

**Ready to explore?** Run: `poetry run python check_setup.py`
