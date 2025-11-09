"""
Setup script for Modal secrets for KURE Project
Run this script to set up necessary secrets for the project
"""

import modal
import os
import getpass

def setup_secrets():
    """Setup Modal secrets for the KURE project"""
    
    print("🔐 Setting up Modal secrets for KURE Project...")
    print("=" * 50)
    
    # Weights & Biases API Key (optional but recommended)
    print("\n1. Weights & Biases Setup (Optional)")
    print("   This is used for experiment tracking and logging.")
    
    wandb_key = input("Enter your Weights & Biases API key (press Enter to skip): ").strip()
    
    if wandb_key:
        try:
            # Create or update the wandb secret
            secret_dict = {"WANDB_API_KEY": wandb_key}
            modal.Secret.from_dict(secret_dict, name="wandb-secret")
            print("✅ Weights & Biases secret created successfully!")
        except Exception as e:
            print(f"❌ Error creating wandb secret: {e}")
    else:
        print("⏭️  Skipping Weights & Biases setup (will run in offline mode)")
    
    # Hugging Face Token (optional, for accessing gated models)
    print("\n2. Hugging Face Token Setup (Optional)")
    print("   This is used for accessing gated models and datasets.")
    
    hf_token = input("Enter your Hugging Face token (press Enter to skip): ").strip()
    
    if hf_token:
        try:
            secret_dict = {"HF_TOKEN": hf_token}
            modal.Secret.from_dict(secret_dict, name="huggingface-secret")
            print("✅ Hugging Face secret created successfully!")
        except Exception as e:
            print(f"❌ Error creating Hugging Face secret: {e}")
    else:
        print("⏭️  Skipping Hugging Face setup")
    
    # OpenAI API Key (optional, for CLIP evaluation)
    print("\n3. OpenAI API Key Setup (Optional)")
    print("   This might be used for advanced evaluation metrics.")
    
    openai_key = input("Enter your OpenAI API key (press Enter to skip): ").strip()
    
    if openai_key:
        try:
            secret_dict = {"OPENAI_API_KEY": openai_key}
            modal.Secret.from_dict(secret_dict, name="openai-secret")
            print("✅ OpenAI secret created successfully!")
        except Exception as e:
            print(f"❌ Error creating OpenAI secret: {e}")
    else:
        print("⏭️  Skipping OpenAI setup")
    
    print("\n" + "=" * 50)
    print("🎉 Modal secrets setup completed!")
    print("\nNext steps:")
    print("1. Deploy the Modal app: modal deploy modal_app.py")
    print("2. Run training: python run_training.py")
    print("3. Generate images: python run_inference.py")

def list_secrets():
    """List existing Modal secrets"""
    try:
        # This is a placeholder - Modal doesn't provide a direct way to list secrets
        print("📋 To view your secrets, check the Modal dashboard at https://modal.com")
    except Exception as e:
        print(f"Error listing secrets: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_secrets()
    else:
        setup_secrets()
