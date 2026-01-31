"""CLI entrypoint for single-point MIA. Run from project root: python run_single_point_mia.py [--config path] [--output dir]."""

from pathlib import Path
import sys

# Bootstrap: add project root's parent to sys.path so "membership_inference_cifar10" is
# the top-level package. Fixes "attempted relative import beyond top-level package" when
# experiments.run_mia does "from ..data import ...".
_root = Path(__file__).resolve().parent
_parent = _root.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import yaml

from membership_inference_cifar10.experiments.run_single_point_mia import run_single_point_mia


def main() -> None:
    import argparse

    root = _root
    ap = argparse.ArgumentParser(description="Single-point MIA: train with/without x, build loss dists, NP threshold, evaluate.")
    ap.add_argument("--config", type=Path, default=root / "configs" / "single_point_mia.yaml", help="Config YAML path")
    ap.add_argument("--data-dir", type=Path, default=root / "datasets", help="CIFAR-100 data root")
    ap.add_argument("--output-dir", type=Path, default=root / "outputs", help="Output directory")
    ap.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    seed = args.seed if args.seed is not None else config.get("seed")
    results = run_single_point_mia(
        config=config,
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        seed=seed,
    )

    # Summary
    eval_ = results["eval"]
    print("\n--- Single-point MIA ---")
    print(f"Target index: {results['target_index']}")
    print(f"Threshold (alpha={results['alpha']}): {results['threshold']:.4f}")
    print(f"Power: {results['power']:.4f}, FPR: {results['fpr']:.4f}")
    # Sanity: over iterations, mean(member loss) should typically be < mean(non-member loss)
    m_mean = float(results["member_losses"].mean())
    n_mean = float(results["non_member_losses"].mean())
    print(f"Mean loss when x IN train (over {len(results['member_losses'])} iters): {m_mean:.4f}")
    print(f"Mean loss when x NOT in train (over {len(results['non_member_losses'])} iters): {n_mean:.4f}")
    print("\nEvaluation (2 held-out models):")
    print(f"  Loss of x when x IN train:    {eval_['loss_when_x_in_train']:.4f}  -> say member? {eval_['say_member_when_x_in_train']}  (correct? {eval_['correct_when_x_in_train']})")
    print(f"  Loss of x when x NOT in train: {eval_['loss_when_x_not_in_train']:.4f}  -> say member? {eval_['say_member_when_x_not_in_train']}  (correct? {eval_['correct_when_x_not_in_train']})")
    print(f"  Both correct: {eval_['both_correct']}")
    print(f"\nHistograms saved to: {args.output_dir / 'single_point_mia_histograms.png'}")


if __name__ == "__main__":
    main()
