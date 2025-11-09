"""
Convenient script to run image generation on Modal
"""

import modal
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate images using NRF on Modal")
    parser.add_argument("--prompt", required=True,
                       help="Text prompt for image generation")
    parser.add_argument("--checkpoint", default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--steps", type=int, default=4,
                       help="Number of sampling steps")
    parser.add_argument("--num-samples", type=int, default=16,
                       help="Number of images to generate")
    parser.add_argument("--output-dir", default="/outputs",
                       help="Output directory for generated images")
    
    args = parser.parse_args()
    
    print(f"🎨 Starting image generation on Modal...")
    print(f"Prompt: '{args.prompt}'")
    print(f"Steps: {args.steps}")
    print(f"Number of samples: {args.num_samples}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    
    # Import the Modal app
    try:
        from modal_app import app, generate_images
    except ImportError:
        print("❌ Error: Could not import modal_app. Make sure modal_app.py is in the same directory.")
        sys.exit(1)
    
    # Run image generation
    with app.run():
        try:
            result = generate_images.remote(
                prompt=args.prompt,
                checkpoint_path=args.checkpoint,
                steps=args.steps,
                num_samples=args.num_samples,
                output_dir=args.output_dir
            )
            
            print("\n" + "=" * 50)
            print("🎉 Image generation completed successfully!")
            print("=" * 50)
            print(f"Status: {result['status']}")
            print(f"Output directory: {result['output_dir']}")
            print(f"Generated files: {len(result['generated_files'])}")
            
            if result['generated_files']:
                print("Generated images:")
                for i, filename in enumerate(result['generated_files'], 1):
                    print(f"  {i}. {filename}")
            
            if result.get('stdout'):
                print("\nDetailed output:")
                print(result['stdout'])
            
        except Exception as e:
            print(f"\n❌ Image generation failed with error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
