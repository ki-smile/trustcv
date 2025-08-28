#!/usr/bin/env python
"""Test that trustcv package builds and imports correctly"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def test_build():
    """Test building the package"""
    print("=" * 60)
    print("TESTING TRUSTCV PACKAGE BUILD")
    print("=" * 60)
    
    # Clean previous builds
    print("\n1. Cleaning previous builds...")
    for dir_name in ['dist', 'build', 'trustcv.egg-info']:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")
    
    # Build the package
    print("\n2. Building package...")
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        return False
    
    print("   ✅ Package built successfully")
    
    # Check created files
    print("\n3. Checking build artifacts...")
    dist_files = list(Path('dist').glob('*'))
    
    expected_files = ['trustcv-1.0.0.tar.gz', 'trustcv-1.0.0-py3-none-any.whl']
    for expected in expected_files:
        if any(expected in str(f) for f in dist_files):
            print(f"   ✅ Found {expected}")
        else:
            print(f"   ❌ Missing {expected}")
    
    # Test installation in temporary environment
    print("\n4. Testing installation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a virtual environment
        venv_dir = Path(tmpdir) / 'test_venv'
        subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
        
        # Get pip from virtual environment
        if sys.platform == 'win32':
            pip_exe = venv_dir / 'Scripts' / 'pip.exe'
            python_exe = venv_dir / 'Scripts' / 'python.exe'
        else:
            pip_exe = venv_dir / 'bin' / 'pip'
            python_exe = venv_dir / 'bin' / 'python'
        
        # Install the wheel
        wheel_file = next(Path('dist').glob('*.whl'))
        result = subprocess.run(
            [str(pip_exe), 'install', str(wheel_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"   ❌ Installation failed:\n{result.stderr}")
            return False
        
        print("   ✅ Package installed successfully")
        
        # Test import
        print("\n5. Testing import...")
        test_code = """
import trustcv
print(f"Version: {trustcv.__version__}")
print(f"Author: {trustcv.__author__}")

# Test importing key components
from trustcv import KFoldMedical, DataLeakageChecker
print("✅ Core components imported successfully")
"""
        
        result = subprocess.run(
            [str(python_exe), '-c', test_code],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"   ❌ Import failed:\n{result.stderr}")
            return False
        
        print(f"   {result.stdout.strip()}")
    
    print("\n" + "=" * 60)
    print("✅ PACKAGE BUILD TEST PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Upload to TestPyPI first:")
    print("   python -m twine upload --repository testpypi dist/*")
    print("\n2. Test installation from TestPyPI:")
    print("   pip install --index-url https://test.pypi.org/simple/ trustcv")
    print("\n3. If everything works, upload to PyPI:")
    print("   python -m twine upload dist/*")
    
    return True

if __name__ == "__main__":
    # Check dependencies
    try:
        import build
        import twine
    except ImportError:
        print("Please install build tools first:")
        print("pip install --upgrade build twine")
        sys.exit(1)
    
    success = test_build()
    sys.exit(0 if success else 1)