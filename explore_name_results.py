"""
Quick exploration script for NAME MVP results.

Run this to see:
- Task difficulty distribution
- Learning curve fits
- Model performance comparison
- Sample learning trajectories
"""

import pandas as pd
import numpy as np
import json

# Load results
learning_curves = pd.read_csv("data/wrangled/name_learning_curves.csv")
horizons = pd.read_csv("data/wrangled/name_horizons.csv")

# Load a sample of runs for detailed inspection
runs_sample = []
with open("data/external/name_runs.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i < 1000:  # First 1000 runs
            runs_sample.append(json.loads(line))
runs_df = pd.DataFrame(runs_sample)

print("=" * 70)
print("NAME MVP EXPLORATION")
print("=" * 70)

# 1. Task Difficulty Overview
print("\n📊 TASK DIFFICULTY DISTRIBUTION")
print("-" * 70)
print(f"Total tasks: {len(learning_curves)}")
print(f"Difficulty range: {learning_curves['first_attempt_success_rate'].min():.3f} - {learning_curves['first_attempt_success_rate'].max():.3f}")
print(f"Mean difficulty: {learning_curves['first_attempt_success_rate'].mean():.3f}")
print(f"Median difficulty: {learning_curves['first_attempt_success_rate'].median():.3f}")
print(f"Std dev: {learning_curves['first_attempt_success_rate'].std():.3f}")

print("\nDifficulty categories:")
easy = (learning_curves['first_attempt_success_rate'] >= 0.6).sum()
medium = ((learning_curves['first_attempt_success_rate'] >= 0.3) &
          (learning_curves['first_attempt_success_rate'] < 0.6)).sum()
hard = (learning_curves['first_attempt_success_rate'] < 0.3).sum()
print(f"  Easy (≥60% first-attempt):   {easy:3d} tasks")
print(f"  Medium (30-60%):             {medium:3d} tasks")
print(f"  Hard (<30%):                 {hard:3d} tasks")

# 2. Learning Curve Parameters
print("\n📈 LEARNING CURVE CHARACTERISTICS")
print("-" * 70)
print(f"Mean learning rate (λ): {learning_curves['fitted_lambda'].mean():.3f}")
print(f"Mean asymptote: {learning_curves['fitted_asymptote'].mean():.3f}")
print(f"Mean RMSE (fit quality): {learning_curves['rmse'].mean():.4f}")

print("\nAttempts needed to reach 50% success:")
finite_n50 = learning_curves[np.isfinite(learning_curves['n_for_50pct'])]
print(f"  Mean: {finite_n50['n_for_50pct'].mean():.2f} attempts")
print(f"  Median: {finite_n50['n_for_50pct'].median():.2f} attempts")
print(f"  Range: {finite_n50['n_for_50pct'].min():.2f} - {finite_n50['n_for_50pct'].max():.2f}")

# 3. Model Performance Comparison
print("\n🤖 AI MODEL PERFORMANCE")
print("-" * 70)
for _, row in horizons.iterrows():
    print(f"\n{row['model']}:")
    print(f"  N-attempt horizon (50%): {row['n_attempt_horizon_50']:.2f} attempts")
    print(f"  N-attempt horizon (80%): {row['n_attempt_horizon_80']:.2f} attempts")
    print(f"  Overall success rate: {row['ai_success_rate_overall']:.1%}")
    print(f"  Difficulty @ 50% AI success: {row['difficulty_at_50pct']:.3f}")
    print(f"  Tasks evaluated: {row['n_tasks']}")

# 4. Example Learning Trajectories
print("\n📚 SAMPLE LEARNING TRAJECTORIES")
print("-" * 70)

# Pick 3 tasks with different difficulties
sample_tasks = learning_curves.sort_values('first_attempt_success_rate').iloc[[10, 50, 90]]

for _, task in sample_tasks.iterrows():
    print(f"\nTask: {task['task_id']}")
    print(f"  First-attempt success: {task['first_attempt_success_rate']:.1%}")
    print(f"  Learning rate (λ): {task['fitted_lambda']:.3f}")
    print(f"  Final asymptote: {task['fitted_asymptote']:.1%}")
    print(f"  Attempts to 50%: {task['n_for_50pct']:.2f}" if np.isfinite(task['n_for_50pct']) else "  Attempts to 50%: N/A (already >50%)")

    # Show predicted success rates over attempts
    base = task['first_attempt_success_rate']
    gain = task['fitted_asymptote'] - base
    lam = task['fitted_lambda']

    print("  Predicted learning curve:")
    for n in [1, 2, 3, 5, 10]:
        p = base + gain * (1 - np.exp(-lam * n))
        print(f"    Attempt {n:2d}: {p:.1%} success")

# 5. Domain Breakdown (if we can extract from task_id)
print("\n🏷️  TASK DOMAINS")
print("-" * 70)
learning_curves['domain'] = learning_curves['task_id'].str.split('/').str[0]
domain_stats = learning_curves.groupby('domain').agg({
    'first_attempt_success_rate': ['count', 'mean'],
    'fitted_lambda': 'mean',
    'fitted_asymptote': 'mean'
})
print(domain_stats.to_string())

# 6. Quick Data Quality Check
print("\n✓ DATA QUALITY CHECKS")
print("-" * 70)
print(f"Learning curves fit successfully: {len(learning_curves)}/100")
print(f"Mean RMSE: {learning_curves['rmse'].mean():.4f} (good if < 0.1)")
print(f"Models evaluated: {len(horizons)}")
print(f"All horizons calculated: {'Yes' if horizons['n_attempt_horizon_50'].notna().all() else 'No'}")

print("\n" + "=" * 70)
print("To visualize these results, see plots/name_*.png")
print("=" * 70)
