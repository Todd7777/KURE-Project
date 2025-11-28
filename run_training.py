"""
Convenient script to run training on Modal
"""

import modal
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run NRF training on Modal")
    parser.add_argument("--config", default="quadratic",
                       help="Configuration name (e.g., 'quadratic', 'spline', 'sb', or a custom name like 'quadratic_coco')")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--wandb-project", default="kure-nrf",
                       help="Weights & Biases project name")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting NRF training on Modal...")
    print(f"Configuration: {args.config}")
    print(f"GPUs: {args.gpus}")
    print(f"W&B Project: {args.wandb_project}")
    
    # Import the Modal app
    try:
        from modal_app import app, train_nrf_model
    except ImportError:
        print("❌ Error: Could not import modal_app. Make sure modal_app.py is in the same directory.")
        sys.exit(1)
    
    # Prepare kwargs
    kwargs = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    
    # Run training
    with app.run():
        try:
            result = train_nrf_model.remote(
                config_name=args.config,
                gpus=args.gpus,
                wandb_project=args.wandb_project,
                **kwargs
            )
            
            print("\n" + "=" * 50)
            print("🎉 Training completed successfully!")
            print("=" * 50)
            print(f"Status: {result['status']}")
            if result.get('stdout'):
                print("Output:")
                print(result['stdout'])
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
