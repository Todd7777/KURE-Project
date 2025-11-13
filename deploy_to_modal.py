#!/usr/bin/env python3
"""
Deployment script for KURE Project on Modal
This script helps deploy the entire KURE project to Modal Labs
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description="", stream_output=False):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        if stream_output:
            # Stream output in real-time - don't capture, just pass through
            # This allows Modal's output to be displayed directly to the user
            process = subprocess.Popen(cmd, stdout=None, stderr=subprocess.STDOUT)
            process.wait()
            if process.returncode == 0:
                print("\n" + "-" * 60)
                print("✅ Success!")
                return True
            else:
                print("\n" + "-" * 60)
                print(f"❌ Error: Command failed with exit code {process.returncode}")
                return False
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Success!")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False

def check_modal_installation():
    """Check if Modal is installed and authenticated"""
    print("🔍 Checking Modal installation...")
    
    # Try python3 -m modal first, then try modal command
    modal_cmd = None
    try:
        result = subprocess.run([sys.executable, "-m", "modal", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal found: {result.stdout.strip()}")
            modal_cmd = [sys.executable, "-m", "modal"]
        else:
            # Try direct modal command
            result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Modal CLI found: {result.stdout.strip()}")
                modal_cmd = ["modal"]
            else:
                print("❌ Modal not found. Please install it:")
                print("   pip install modal")
                return False
    except FileNotFoundError:
        print("❌ Modal not found. Please install it:")
        print("   pip install modal")
        return False
    
    # Check if we can import Modal (basic check)
    try:
        import modal
        print(f"✅ Modal Python package installed: {modal.__version__}")
    except ImportError:
        print("❌ Modal Python package not found. Please install it:")
        print("   pip install modal")
        return False
    
    # Store modal command for later use
    check_modal_installation.modal_cmd = modal_cmd
    print("✅ Modal installation verified")
    return True

def deploy_main_app():
    """Deploy the main Modal app"""
    print("\n" + "="*50)
    print("🚀 DEPLOYING MAIN MODAL APP")
    print("="*50)
    
    modal_cmd = getattr(check_modal_installation, 'modal_cmd', [sys.executable, "-m", "modal"])
    # Use stream-logs to see build output in real-time
    return run_command(
        modal_cmd + ["deploy", "modal_app.py", "--stream-logs"],
        "Deploying main KURE Modal app...",
        stream_output=True
    )

def deploy_web_interface():
    """Deploy the web interface"""
    print("\n" + "="*50)
    print("🌐 DEPLOYING WEB INTERFACE")
    print("="*50)
    
    modal_cmd = getattr(check_modal_installation, 'modal_cmd', [sys.executable, "-m", "modal"])
    return run_command(
        modal_cmd + ["deploy", "modal_web_interface.py", "--stream-logs"],
        "Deploying web interface...",
        stream_output=True
    )

def setup_secrets():
    """Setup Modal secrets"""
    print("\n" + "="*50)
    print("🔐 SETTING UP SECRETS")
    print("="*50)
    
    response = input("Do you want to set up API secrets now? (y/n): ").lower().strip()
    if response == 'y':
        return run_command(
            [sys.executable, "setup_modal_secrets.py"],
            "Setting up Modal secrets..."
        )
    else:
        print("⏭️  Skipping secrets setup. You can run 'python setup_modal_secrets.py' later.")
        return True

def test_deployment():
    """Test the deployment"""
    print("\n" + "="*50)
    print("🧪 TESTING DEPLOYMENT")
    print("="*50)
    
    print("Testing basic functionality...")
    
    # Test listing configs
    modal_cmd = getattr(check_modal_installation, 'modal_cmd', [sys.executable, "-m", "modal"])
    test_cmd = modal_cmd + ["run", "modal_app.py::list_available_configs"]
    
    return run_command(test_cmd, "Testing configuration listing...")

def main():
    """Main deployment function"""
    print("🎨 KURE Project Modal Deployment Script")
    print("="*50)
    
    # Check current directory
    if not os.path.exists("modal_app.py"):
        print("❌ Error: modal_app.py not found in current directory")
        print("Please run this script from the KURE-Project directory")
        sys.exit(1)
    
    # Check Modal installation
    if not check_modal_installation():
        sys.exit(1)
    
    # Setup secrets first (optional) - skip interactively for automation
    print("⏭️  Skipping secrets setup (optional). You can run 'python setup_modal_secrets.py' later.")
    
    # Uncomment below if you want interactive secrets setup:
    # if not setup_secrets():
    #     print("⚠️  Warning: Secrets setup failed, but continuing with deployment...")
    
    # Deploy main app
    if not deploy_main_app():
        print("❌ Main app deployment failed!")
        sys.exit(1)
    
    # Deploy web interface
    if not deploy_web_interface():
        print("⚠️  Warning: Web interface deployment failed, but main app is deployed")
    
    # Test deployment
    if not test_deployment():
        print("⚠️  Warning: Deployment test failed, but apps are deployed")
    
    print("\n" + "="*60)
    print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. 📋 Check your Modal dashboard: https://modal.com/apps")
    print("2. 🎨 Try image generation:")
    print("   python run_inference.py --prompt 'A beautiful sunset'")
    print("3. 🚂 Start training:")
    print("   python run_training.py --config quadratic")
    print("4. 🌐 Access web interface from Modal dashboard")
    print("\n📚 For detailed usage, see MODAL_DEPLOYMENT.md")

if __name__ == "__main__":
    main()
