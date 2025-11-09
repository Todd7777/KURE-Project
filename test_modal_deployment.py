"""
Test script for Modal deployment
This script tests the basic functionality of the deployed Modal app
"""

import modal
import sys
import time

def test_basic_functionality():
    """Test basic Modal app functionality"""
    print("🧪 Testing Modal deployment...")
    
    try:
        # Import the Modal app
        from modal_app import app, list_available_configs, generate_images
        
        print("✅ Successfully imported Modal app")
        
        # Test 1: List available configurations
        print("\n📋 Test 1: Listing available configurations...")
        with app.run():
            configs = list_available_configs.remote()
            print(f"✅ Available configs: {configs}")
        
        # Test 2: Quick image generation (small test)
        print("\n🎨 Test 2: Quick image generation test...")
        with app.run():
            result = generate_images.remote(
                prompt="A simple red circle",
                steps=1,  # Minimal steps for quick test
                num_samples=1  # Just one image
            )
            
            if result['status'] == 'success':
                print(f"✅ Image generation successful!")
                print(f"   Generated {len(result['generated_files'])} image(s)")
                print(f"   Output directory: {result['output_dir']}")
            else:
                print(f"❌ Image generation failed: {result}")
                return False
        
        print("\n🎉 All tests passed! Modal deployment is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure modal_app.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_web_interface():
    """Test web interface deployment"""
    print("\n🌐 Testing web interface...")
    
    try:
        from modal_web_interface import app as web_app
        print("✅ Web interface app imported successfully")
        
        # The web interface test would require actually deploying and accessing the endpoint
        # For now, we just verify it can be imported
        print("✅ Web interface appears to be configured correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Web interface import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def test_runner_scripts():
    """Test that runner scripts are properly configured"""
    print("\n🎮 Testing runner scripts...")
    
    scripts_to_test = [
        "run_training.py",
        "run_inference.py", 
        "run_evaluation.py",
        "setup_modal_secrets.py"
    ]
    
    import os
    
    for script in scripts_to_test:
        if os.path.exists(script):
            print(f"✅ {script} found")
        else:
            print(f"❌ {script} not found")
            return False
    
    print("✅ All runner scripts are present")
    return True

def main():
    """Run all tests"""
    print("🚀 KURE Project Modal Deployment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Runner Scripts", test_runner_scripts),
        ("Web Interface", test_web_interface), 
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Modal deployment is ready to use.")
        print("\nNext steps:")
        print("1. Deploy to Modal: python deploy_to_modal.py")
        print("2. Generate images: python run_inference.py --prompt 'A beautiful sunset'")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
