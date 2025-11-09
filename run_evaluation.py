"""
Convenient script to run model evaluation on Modal
"""

import modal
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Evaluate NRF model on Modal")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", default="coco",
                       choices=["coco", "compositional"],
                       help="Dataset to evaluate on")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8],
                       help="List of sampling steps to evaluate")
    parser.add_argument("--metrics", nargs="+", default=["fid", "is", "clip"],
                       choices=["fid", "is", "clip"],
                       help="List of metrics to compute")
    
    args = parser.parse_args()
    
    print(f"📊 Starting model evaluation on Modal...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps: {args.steps}")
    print(f"Metrics: {args.metrics}")
    
    # Import the Modal app
    try:
        from modal_app import app, evaluate_model
    except ImportError:
        print("❌ Error: Could not import modal_app. Make sure modal_app.py is in the same directory.")
        sys.exit(1)
    
    # Run evaluation
    with app.run():
        try:
            result = evaluate_model.remote(
                checkpoint_path=args.checkpoint,
                dataset=args.dataset,
                steps=args.steps,
                metrics=args.metrics
            )
            
            print("\n" + "=" * 50)
            print("🎉 Evaluation completed successfully!")
            print("=" * 50)
            print(f"Status: {result['status']}")
            
            if result.get('stdout'):
                print("Evaluation results:")
                print(result['stdout'])
            
        except Exception as e:
            print(f"\n❌ Evaluation failed with error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
