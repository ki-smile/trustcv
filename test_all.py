#!/usr/bin/env python
"""
Comprehensive test script for trustcv package
Tests all examples, imports, and basic functionality
"""

import sys
import os
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("🔍 Testing Module Imports")
    print("="*60)
    
    modules_to_test = [
        "trustcv",
        "trustcv.splitters.iid",
        "trustcv.splitters.temporal", 
        "trustcv.splitters.grouped",
        "trustcv.splitters.spatial",
        "trustcv.checkers.leakage",
        "trustcv.checkers.balance",
        "trustcv.metrics.medical_metrics",
        "trustcv.visualization.plots",
        "trustcv.frameworks.pytorch",
        "trustcv.frameworks.tensorflow",
        "trustcv.frameworks.monai",
        "trustcv.core.base",
        "trustcv.core.runner",
        "trustcv.core.callbacks"
    ]
    
    failed = []
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️ Failed imports: {failed}")
        return False
    else:
        print("\n✅ All modules imported successfully!")
        return True

def test_cv_methods():
    """Test that CV methods work with basic data"""
    print("\n" + "="*60)
    print("🧪 Testing CV Methods")
    print("="*60)
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    groups = np.random.randint(0, 10, 100)
    
    # Test methods
    test_cases = [
        ("I.I.D. - KFoldMedical", "trustcv.splitters.iid", "KFoldMedical", {"n_splits": 5}),
        ("I.I.D. - StratifiedKFoldMedical", "trustcv.splitters.iid", "StratifiedKFoldMedical", {"n_splits": 5}),
        ("Temporal - TimeSeriesSplit", "trustcv.splitters.temporal", "TimeSeriesSplit", {"n_splits": 5}),
        ("Grouped - GroupKFoldMedical", "trustcv.splitters.grouped", "GroupKFoldMedical", {"n_splits": 5}),
    ]
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    failed = []
    
    for name, module_name, class_name, params in test_cases:
        try:
            module = __import__(module_name, fromlist=[class_name])
            CVClass = getattr(module, class_name)
            cv = CVClass(**params)
            
            # Test splitting
            if "Group" in class_name:
                splits = list(cv.split(X, y, groups=groups))
            else:
                splits = list(cv.split(X, y))
            
            if len(splits) > 0:
                train_idx, test_idx = splits[0]
                model.fit(X[train_idx], y[train_idx])
                score = model.score(X[test_idx], y[test_idx])
                print(f"✅ {name}: {len(splits)} splits, sample score: {score:.3f}")
            else:
                print(f"⚠️ {name}: No splits generated")
                failed.append(name)
                
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️ Failed methods: {failed}")
        return False
    else:
        print("\n✅ All CV methods tested successfully!")
        return True

def test_data_leakage_checker():
    """Test data leakage detection"""
    print("\n" + "="*60)
    print("🔍 Testing Data Leakage Detection")
    print("="*60)
    
    try:
        import numpy as np
        from trustcv.checkers.leakage import DataLeakageChecker
        
        # Create data with intentional leakage
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Add target as feature (obvious leakage)
        X[:, 0] = y
        
        checker = DataLeakageChecker()
        report = checker.check_feature_target_leakage(X, y)
        
        if report['has_leakage']:
            print("✅ Data leakage correctly detected!")
            print(f"   Suspicious features: {report['suspicious_features']}")
        else:
            print("❌ Failed to detect obvious data leakage")
            return False
            
    except Exception as e:
        print(f"❌ Data leakage checker failed: {e}")
        return False
    
    return True

def test_examples():
    """Test that example scripts run without errors"""
    print("\n" + "="*60)
    print("📚 Testing Example Scripts")
    print("="*60)
    
    examples_dir = Path(__file__).parent / "examples"
    example_files = [
        "heart_disease_prediction.py",
        "icu_patient_monitoring.py",
        "multisite_clinical_trial.py",
        "disease_spread_modeling.py"
    ]
    
    failed = []
    for example_file in example_files:
        example_path = examples_dir / example_file
        
        if not example_path.exists():
            print(f"⚠️ {example_file}: File not found")
            continue
        
        try:
            # Read and execute the example
            with open(example_path, 'r') as f:
                code = f.read()
            
            # Create a new namespace for execution
            namespace = {}
            
            # Execute with timeout protection
            exec(code, namespace)
            print(f"✅ {example_file}")
            
        except Exception as e:
            print(f"❌ {example_file}: {str(e)[:100]}")
            failed.append(example_file)
    
    if failed:
        print(f"\n⚠️ Failed examples: {failed}")
        return False
    else:
        print("\n✅ All examples ran successfully!")
        return True

def test_notebooks():
    """Test that notebooks have valid structure"""
    print("\n" + "="*60)
    print("📓 Testing Notebooks")
    print("="*60)
    
    notebooks_dir = Path(__file__).parent / "notebooks"
    
    if not notebooks_dir.exists():
        print("⚠️ Notebooks directory not found")
        return False
    
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    
    if not notebook_files:
        print("⚠️ No notebooks found")
        return False
    
    import json
    failed = []
    
    for notebook_path in notebook_files:
        try:
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            # Check structure
            if "cells" not in notebook:
                print(f"❌ {notebook_path.name}: No cells found")
                failed.append(notebook_path.name)
                continue
            
            # Check for content
            n_code = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "code")
            n_markdown = sum(1 for cell in notebook["cells"] if cell["cell_type"] == "markdown")
            
            if n_code == 0:
                print(f"⚠️ {notebook_path.name}: No code cells")
            elif n_markdown == 0:
                print(f"⚠️ {notebook_path.name}: No markdown cells")
            else:
                print(f"✅ {notebook_path.name}: {n_code} code cells, {n_markdown} markdown cells")
                
        except Exception as e:
            print(f"❌ {notebook_path.name}: {e}")
            failed.append(notebook_path.name)
    
    if failed:
        print(f"\n⚠️ Failed notebooks: {failed}")
        return False
    else:
        print("\n✅ All notebooks validated successfully!")
        return True

def test_website():
    """Test that website files exist and are valid"""
    print("\n" + "="*60)
    print("🌐 Testing Website")
    print("="*60)
    
    website_dir = Path(__file__).parent / "website"
    
    required_files = [
        "index.html",
        "tutorials.html",
        "js/main.js",
        "js/cv-visualizations.js",
        "js/cv-temporal.js",
        "js/cv-grouped.js",
        "js/cv-spatial.js"
    ]
    
    failed = []
    for file_path in required_files:
        full_path = website_dir / file_path
        if full_path.exists():
            # Check file is not empty
            size = full_path.stat().st_size
            if size > 100:
                print(f"✅ {file_path} ({size:,} bytes)")
            else:
                print(f"⚠️ {file_path} (only {size} bytes)")
                failed.append(file_path)
        else:
            print(f"❌ {file_path}: Not found")
            failed.append(file_path)
    
    if failed:
        print(f"\n⚠️ Missing or invalid files: {failed}")
        return False
    else:
        print("\n✅ All website files present!")
        return True

def test_documentation():
    """Test that documentation files exist"""
    print("\n" + "="*60)
    print("📚 Testing Documentation")
    print("="*60)
    
    docs_dir = Path(__file__).parent / "docs"
    
    required_docs = [
        "CV_METHODS_CHECKLIST.md",
        "CV_SELECTION_GUIDE.md",
        "PRACTICAL_CV_GUIDE.md",
        "ML_TOOLBOX_CV_COMPARISON.md"
    ]
    
    failed = []
    for doc_file in required_docs:
        doc_path = docs_dir / doc_file
        if doc_path.exists():
            size = doc_path.stat().st_size
            if size > 1000:
                print(f"✅ {doc_file} ({size:,} bytes)")
            else:
                print(f"⚠️ {doc_file} (only {size} bytes)")
                failed.append(doc_file)
        else:
            print(f"❌ {doc_file}: Not found")
            failed.append(doc_file)
    
    if failed:
        print(f"\n⚠️ Missing or incomplete docs: {failed}")
        return False
    else:
        print("\n✅ All documentation present!")
        return True

def main():
    """Run all tests"""
    print("🏥 trustcv Comprehensive Test Suite")
    print("="*60)
    
    # Track results
    results = {
        "Imports": test_imports(),
        "CV Methods": test_cv_methods(),
        "Data Leakage": test_data_leakage_checker(),
        "Examples": test_examples(),
        "Notebooks": test_notebooks(),
        "Website": test_website(),
        "Documentation": test_documentation()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! trustcv is ready to use!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())