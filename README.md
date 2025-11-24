# AI Capability Evaluation: Time-Horizon & Learning Difficulty Metrics

This repository contains two complementary approaches to measuring AI capabilities in human-interpretable terms:

1. **Time-Horizon Metric** (Original METR work) - Measures AI capability by task completion time
2. **NAME (Novelty Achievement MeasurE)** (New experimental framework) - Measures AI capability by learning difficulty

## Repository Contents

### Time-Horizon Evaluation (Original)

This is the code for the paper *Measuring AI Ability to Complete Long Tasks* by METR.

Despite rapid progress on AI benchmarks, the real-world meaning of benchmark performance remains unclear. To quantify the capabilities of AI systems in terms of human capabilities, we propose a new metric: **50%-task-completion time horizon**. This is the time humans typically take to complete tasks that AI models can complete with 50% success rate.

We first timed humans with relevant domain expertise on a combination of RE-Bench, HCAST, and 66 novel shorter tasks. On these tasks, current frontier AI models such as Claude 3.7 Sonnet have a 50% time horizon of around 50 minutes. Furthermore, frontier AI time horizon has been doubling approximately every seven months since 2019, though the trend may have accelerated in 2024.

**Key findings**: If these results generalize to real-world software tasks, extrapolation predicts that within 5 years, AI systems will be capable of automating many software tasks that currently take humans a month.

**Paper**: https://arxiv.org/pdf/2503.14499

### NAME Evaluation (New Experimental Framework)

**NAME (Novelty Achievement MeasurE)** is a novel evaluation metric that measures AI capability by comparing AI zero-shot performance against novice human learning curves.

**Core Innovation**: Instead of asking "how long do tasks take experts?", NAME asks: **"Can AI solve on first try what novices learn over multiple attempts?"**

**Key Metric**: **N-Attempt Equivalence Horizon** - The task difficulty (measured by attempts novices need) where AI achieves target success rate.

**Example**: "GPT-4 has a 3-attempt equivalence" means it can zero-shot solve tasks that novices need 3 attempts to reach 50% success.

**Status**: 🚧 **MVP Complete with Synthetic Data** - Ready for real human data collection

## What Makes NAME Unique?

NAME is the first systematic benchmark combining:

1. **Novice-Centric Baseline**: Compares to novice first-attempt, not expert performance
2. **Learning Curve Comparison**: Captures full learning trajectories over multiple attempts
3. **Everyday Task Focus**: Targets common tasks accessible to novices (when scaled up)
4. **Interpretable Metric**: N-attempt equivalence is intuitive and trackable over time

**Comparison to Time-Horizon**:

| Aspect | Time Horizon | NAME |
|--------|-------------|------|
| **Question** | "How long do experts take?" | "How hard to learn?" |
| **Baseline** | Domain experts | Task-specific novices |
| **Measurement** | Completion time | Attempts to proficiency |
| **Task type** | Expert-level, verifiable | Everyday, novice-accessible |
| **Learning signal** | No (single attempt) | Yes (learning curves) |

Both metrics are complementary and provide different insights into AI capabilities.

## Quick Start

### Option 1: Explore NAME (New Framework)

**Setup & Verification:**
```bash
# Install dependencies
poetry install

# Verify setup
poetry run python check_setup.py
```

**Generate NAME evaluation data:**
```bash
# Run complete pipeline (generates 56,000 runs, ~2-3 minutes)
poetry run dvc repro plot_name_results
```

**Explore results:**
```bash
# Quick overview (command line)
poetry run python explore_name_results.py

# Interactive exploration (Jupyter)
poetry run jupyter notebook explore_name_interactive.ipynb

# View plots
open plots/name_*.png
```

**See detailed documentation:**
- **[QUICKSTART.md](QUICKSTART.md)** - Setup guide with troubleshooting
- **[docs/NAME_README.md](docs/NAME_README.md)** - Complete NAME documentation (320+ lines)
- **[docs/NAME_IMPLEMENTATION_PLAN.md](docs/NAME_IMPLEMENTATION_PLAN.md)** - Architecture & scale-up plan

### Option 2: Reproduce Time-Horizon Results (Original)

**Setup:**
```bash
poetry install
poetry run dvc repro
```

**Explore results:**
```bash
jupyter notebook example_analysis.ipynb
```

## NAME MVP Architecture

```
Synthetic Task Generator (100 tasks, 5 domains)
    ↓
Novice Learning Simulator (50 learners × 10 attempts)
    ↓
AI Zero-Shot Simulator (3 capability levels)
    ↓
56,000 Synthetic Runs
    ↓
Learning Curve Fitter (exponential model)
    ↓
N-Attempt Horizon Calculator
    ↓
Visualizations & Metrics
```

**Current Results (Synthetic Data)**:

| AI Model | N-Attempt Horizon (50%) | Interpretation |
|----------|------------------------|----------------|
| baseline-weak | 1.16 attempts | Below novice first-attempt |
| baseline-medium | 0.38 attempts | Near novice first-attempt |
| baseline-strong | 0.31 attempts | Above novice first-attempt |

## NAME Goals & Roadmap

### Research Goals

1. **Develop learning-based evaluation**: Create systematic framework for comparing AI zero-shot performance to human learning curves
2. **Validate with real data**: Collect human novice data (N=100+ participants) and validate learning curve models
3. **Track AI progress**: Establish baseline NAME horizons for frontier models and track over time
4. **Predict consumer adoption**: Test whether NAME better predicts real-world AI adoption than expert-focused benchmarks

### Technical Goals

1. **✅ MVP with synthetic data** (COMPLETE)
   - Synthetic task generator across 5 domains
   - Novice learning simulator with exponential model
   - NAME metric calculation with curve fitting
   - DVC pipeline for reproducibility
   - Comprehensive visualizations

2. **⬜ Real human data collection** (NEXT)
   - Recruit task-specific novices (N=100+ per task type)
   - Track learning over 10 attempts with binary feedback
   - Validate learning curve models against real data
   - Collect baseline for 10+ task types

3. **⬜ Real AI model evaluation**
   - Integrate GPT-4, Claude, o1, and other frontier models
   - Run zero-shot evaluations on validated tasks
   - Calculate NAME horizons for real models
   - Compare to time-horizon metrics

4. **⬜ Broad domain expansion**
   - Household tasks (cooking, home repair, organization)
   - Creative tasks (writing, art, storytelling)
   - Social tasks (communication, conflict resolution)
   - Professional tasks (data entry, scheduling, email)
   - Target 100+ tasks across 10+ domains

5. **⬜ Advanced analysis**
   - Few-shot learning comparison (AI N-shot vs human N-attempt)
   - Transfer learning measurement
   - Failure mode analysis
   - Skill composition decomposition

### Scale-Up Criteria

**Phase 1 → Phase 2** (MVP → Real Data):
- ✅ MVP demonstrates clear signal in synthetic data
- ✅ Stakeholder buy-in and interest
- ⬜ Budget allocated for human data collection ($10-50K)
- ⬜ IRB approval obtained (if required)
- ⬜ Recruitment platform identified

**Phase 2 → Phase 3** (Real Data → Broad Expansion):
- ⬜ Real data collection validated (N>50 participants, 3+ task types)
- ⬜ Inter-rater reliability established (κ > 0.7)
- ⬜ Learning curves fit real data well (RMSE < 0.1)
- ⬜ Initial results published or presented

**Phase 3 → Phase 4** (Expansion → Advanced Analysis):
- ⬜ 10+ domains with validated tasks
- ⬜ Baseline NAME horizons for 5+ frontier models
- ⬜ Longitudinal data showing trend over time
- ⬜ Community interest confirmed (citations, adoption)

## Installation

### Using Dev Container (Recommended)

This project contains a dev container configuration. See [.devcontainer/Dockerfile](.devcontainer/Dockerfile) for dependencies.

### Manual Installation

```bash
# Install dependencies
poetry install

# Verify setup (for NAME)
poetry run python check_setup.py

# Run time-horizon pipeline
poetry run dvc repro

# Run NAME pipeline
poetry run dvc repro plot_name_results
```

### Dependencies

Main packages:
- Python 3.11+
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- pyyaml
- DVC (Data Version Control)

See `pyproject.toml` for complete list.

## Project Structure

```
eval-analysis-public_exp/
│
├── README.md                           # This file
├── QUICKSTART.md                       # NAME quick start guide
│
├── docs/
│   ├── NAME_README.md                  # Complete NAME documentation
│   └── NAME_IMPLEMENTATION_PLAN.md     # Architecture & roadmap
│
├── src/
│   ├── generate/                       # NAME: Data generation
│   │   ├── synthetic_tasks.py          # Task generator
│   │   └── generate_runs.py            # Run simulator
│   ├── simulate/                       # NAME: Learning models
│   │   └── novice_learning.py          # Exponential learning model
│   ├── wrangle/                        # Both: Metric calculation
│   │   ├── name_metrics.py             # NAME metric calculator
│   │   ├── logistic.py                 # Time-horizon calculator
│   │   └── ...
│   ├── plot/                           # Both: Visualizations
│   │   ├── name_plots.py               # NAME visualizations
│   │   └── ...
│   └── utils/                          # Shared utilities
│
├── data/
│   ├── external/
│   │   ├── all_runs.jsonl              # Time-horizon: 18,964 runs
│   │   ├── name_tasks.yaml             # NAME: 100 synthetic tasks
│   │   └── name_runs.jsonl             # NAME: 56,000 runs
│   └── wrangled/
│       ├── logistic_fits/              # Time-horizon: Fitted curves
│       ├── name_learning_curves.csv    # NAME: Learning curves
│       └── name_horizons.csv           # NAME: N-attempt horizons
│
├── plots/                              # Generated visualizations
│   ├── name_*.png                      # NAME plots (4 files)
│   └── *.png                           # Time-horizon plots
│
├── explore_name_results.py             # NAME: Quick exploration
├── explore_name_interactive.ipynb      # NAME: Interactive notebook
├── check_setup.py                      # NAME: Setup verification
│
├── example_analysis.ipynb              # Time-horizon: Example analysis
│
├── dvc.yaml                            # Pipeline definitions (both)
└── pyproject.toml                      # Dependencies
```

## Example Outputs

### NAME Visualizations

**Learning Curves** - Shows novice learning over 10 attempts with exponential model fits

**AI vs Human First-Attempt** - Compares AI zero-shot to novice first-attempt success across difficulty spectrum

**N-Attempt Horizons** - Bar charts showing the core NAME metric for each AI model

**Difficulty Distribution** - Histogram of task difficulties

See `plots/name_*.png` after running the pipeline.

### Time-Horizon Visualizations

See original plots in `plots/` directory after running `dvc repro`.

## Use Cases

### For Researchers

**Time-Horizon Metric:**
- Measuring AI progress on expert-level tasks
- Predicting automation of specialized work
- Understanding AI capability growth trends

**NAME Metric:**
- Measuring AI learning efficiency vs humans
- Predicting consumer AI adoption
- Understanding AI performance on everyday tasks
- Developing learning-based benchmarks

### For Practitioners

**Evaluate your AI models:**
1. Run zero-shot evaluations on NAME tasks
2. Compare to baseline horizons
3. Track progress over time
4. Identify capability gaps

**Collect human baselines:**
1. Use NAME task format
2. Track novice learning over attempts
3. Fit learning curves
4. Calculate N-attempt thresholds

## Contributing

We welcome contributions! Areas of interest:

- **Real human data collection** - Help design and run studies
- **New task domains** - Propose and validate everyday tasks
- **AI model integration** - Add API adapters for new models
- **Learning models** - Test alternative curve-fitting approaches
- **Analysis tools** - Improve visualizations and metrics

## Citation

### Time-Horizon Metric

```bibtex
@article{metr2025timehorizon,
  title={Measuring AI Ability to Complete Long Tasks},
  author={[METR Authors]},
  year={2025},
  url={https://arxiv.org/pdf/2503.14499}
}
```

### NAME Metric

```bibtex
@software{name2025,
  title={NAME: Novelty Achievement MeasurE - A Learning-Based AI Evaluation Framework},
  author={[Your Name]},
  year={2025},
  note={Experimental framework, MVP with synthetic data},
  url={https://github.com/[your-repo]/eval-analysis-public_exp}
}
```

## License

[Same as original METR repository]

## Contact

- **Time-Horizon Questions**: See original METR paper
- **NAME Questions**: [Your contact]
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Status**:
- Time-Horizon: ✅ Published results with real data
- NAME: 🚧 MVP complete with synthetic data, ready for real data collection

**Last Updated**: 2025-11-24
