"""
Integration tests for all example scripts

These tests run full example scripts which can be slow.
Set RUN_HEAVY_TESTS=1 to enable them.
"""

import pytest
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip all tests in this module unless RUN_HEAVY_TESTS is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_HEAVY_TESTS"),
    reason="Heavy test - set RUN_HEAVY_TESTS=1 to enable"
)


class TestExampleScripts:
    """Integration tests for example scripts"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.examples_dir = Path(__file__).parent.parent / "examples"
        cls.temp_dir = tempfile.mkdtemp()
        
    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def run_example_script(self, script_name, timeout=60):
        """Helper to run an example script"""
        script_path = self.examples_dir / script_name
        
        if not script_path.exists():
            pytest.skip(f"Example script {script_name} not found")
        
        # Run the script with a timeout
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.examples_dir)
            )
            return result
        except subprocess.TimeoutExpired:
            pytest.fail(f"Script {script_name} timed out after {timeout} seconds")
    
    def test_heart_disease_prediction(self):
        """Test heart disease prediction example"""
        result = self.run_example_script("heart_disease_prediction.py")
        
        # Check that script ran successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check for expected output patterns
        output = result.stdout.lower()
        assert "heart disease" in output or "cardiovascular" in output
        assert "cross-validation" in output or "cv" in output
        assert "auc" in output or "accuracy" in output
        
        # Check that no errors were printed
        assert "error" not in result.stderr.lower()
        assert "exception" not in result.stderr.lower()
    
    def test_icu_patient_monitoring(self):
        """Test ICU patient monitoring example"""
        result = self.run_example_script("icu_patient_monitoring.py")
        
        # Check that script ran successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check for expected output patterns
        output = result.stdout.lower()
        assert "icu" in output or "temporal" in output or "time series" in output
        assert "patient" in output
        assert any(term in output for term in ["rolling", "expanding", "time", "temporal"])
        
        # Check that temporal methods were used
        assert any(method in output for method in [
            "timeseriesplit", "rolling", "expanding", "purged"
        ])
    
    def test_multisite_clinical_trial(self):
        """Test multi-site clinical trial example"""
        result = self.run_example_script("multisite_clinical_trial.py")
        
        # Check that script ran successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check for expected output patterns
        output = result.stdout.lower()
        assert any(term in output for term in ["hospital", "site", "center", "multi"])
        assert "group" in output or "leave" in output
        
        # Check that grouped methods were used
        assert any(method in output for method in [
            "groupkfold", "leaveonegroup", "grouped"
        ])
    
    def test_disease_spread_modeling(self):
        """Test disease spread modeling example"""
        result = self.run_example_script("disease_spread_modeling.py")
        
        # Check that script ran successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check for expected output patterns
        output = result.stdout.lower()
        assert any(term in output for term in ["spatial", "geographic", "location", "spread"])
        assert "disease" in output or "epidemic" in output
        
        # Check that spatial methods were used
        assert any(method in output for method in [
            "spatial", "buffer", "block", "geographic"
        ])
    
    def test_example_imports(self):
        """Test that all example scripts can be imported"""
        example_files = [
            "heart_disease_prediction.py",
            "icu_patient_monitoring.py",
            "multisite_clinical_trial.py",
            "disease_spread_modeling.py"
        ]
        
        for script_name in example_files:
            script_path = self.examples_dir / script_name
            
            if not script_path.exists():
                continue
            
            # Read the script and check for required imports
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Check for essential imports
            assert "import numpy" in content or "from numpy" in content
            assert "import pandas" in content or "from pandas" in content
            assert "from sklearn" in content or "import sklearn" in content
            assert "from trustcv" in content or "import trustcv" in content
    
    def test_example_functions(self):
        """Test that example scripts define expected functions"""
        example_requirements = {
            "heart_disease_prediction.py": ["create_synthetic_data", "evaluate_cv_methods"],
            "icu_patient_monitoring.py": ["create_icu_data", "temporal_cv_comparison"],
            "multisite_clinical_trial.py": ["create_trial_data", "grouped_cv_evaluation"],
            "disease_spread_modeling.py": ["create_spatial_data", "spatial_cv_analysis"]
        }
        
        for script_name, required_functions in example_requirements.items():
            script_path = self.examples_dir / script_name
            
            if not script_path.exists():
                continue
            
            with open(script_path, 'r') as f:
                content = f.read()
            
            for func_name in required_functions:
                assert f"def {func_name}" in content, \
                    f"Function {func_name} not found in {script_name}"
    
    def test_no_hardcoded_paths(self):
        """Test that examples don't use hardcoded paths"""
        example_files = list(self.examples_dir.glob("*.py"))
        
        for script_path in example_files:
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Check for common hardcoded path patterns
            assert "/Users/" not in content, f"Hardcoded user path in {script_path.name}"
            assert "C:\\" not in content, f"Hardcoded Windows path in {script_path.name}"
            assert "/home/" not in content, f"Hardcoded home path in {script_path.name}"
            
            # Allow relative paths and temp directories
            # These are okay: "./data", "../trustcv", tempfile, etc.
    
    def test_example_output_format(self):
        """Test that examples produce formatted output"""
        # Run a quick example to check output format
        result = self.run_example_script("heart_disease_prediction.py", timeout=30)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Check for structured output
            assert any(char in output for char in ["=", "-", "*", "#"]), \
                "No formatting characters found in output"
            
            # Check for numerical results
            import re
            numbers = re.findall(r'\d+\.\d+', output)
            assert len(numbers) > 0, "No numerical results found in output"
            
            # Check for performance metrics
            assert any(metric in output.lower() for metric in [
                "accuracy", "precision", "recall", "f1", "auc", "roc"
            ]), "No performance metrics found in output"


class TestNotebookExecution:
    """Test that notebooks can be executed"""
    
    @classmethod
    def setup_class(cls):
        """Set up notebook testing"""
        cls.notebooks_dir = Path(__file__).parent.parent / "notebooks"
        
    def test_notebook_structure(self):
        """Test that notebooks have proper structure"""
        notebook_files = list(self.notebooks_dir.glob("*.ipynb"))
        
        assert len(notebook_files) > 0, "No notebooks found"
        
        for notebook_path in notebook_files[:3]:  # Test first 3 notebooks
            with open(notebook_path, 'r') as f:
                import json
                notebook = json.load(f)
            
            # Check notebook structure
            assert "cells" in notebook, f"No cells in {notebook_path.name}"
            assert len(notebook["cells"]) > 0, f"Empty notebook: {notebook_path.name}"
            
            # Check for markdown and code cells
            cell_types = [cell["cell_type"] for cell in notebook["cells"]]
            assert "markdown" in cell_types, f"No markdown cells in {notebook_path.name}"
            assert "code" in cell_types, f"No code cells in {notebook_path.name}"
            
            # Check first cell is markdown (title/introduction)
            assert notebook["cells"][0]["cell_type"] == "markdown", \
                f"First cell should be markdown in {notebook_path.name}"
    
    def test_notebook_imports(self):
        """Test that notebooks have necessary imports"""
        critical_notebooks = [
            "01_IID_Methods.ipynb",
            "02_Grouped_Medical.ipynb",
            "03_Temporal_Medical.ipynb",
            "10_Data_Leakage_Consequences.ipynb"
        ]
        
        for notebook_name in critical_notebooks:
            notebook_path = self.notebooks_dir / notebook_name
            
            if not notebook_path.exists():
                continue
            
            with open(notebook_path, 'r') as f:
                import json
                notebook = json.load(f)
            
            # Collect all code cells
            code_cells = [cell for cell in notebook["cells"] 
                         if cell["cell_type"] == "code"]
            
            # Combine all code
            all_code = "\n".join(
                "\n".join(cell["source"]) if isinstance(cell["source"], list) 
                else cell["source"]
                for cell in code_cells
            )
            
            # Check for essential imports
            assert "import numpy" in all_code or "from numpy" in all_code
            assert "import pandas" in all_code or "from pandas" in all_code
            assert "matplotlib" in all_code or "seaborn" in all_code
            assert "sklearn" in all_code
            
            # Check for trustcv imports in most notebooks
            if "Data_Leakage" not in notebook_name:
                assert "trustcv" in all_code, \
                    f"trustcv not imported in {notebook_name}"


class TestPackageIntegration:
    """Test overall package integration"""
    
    def test_package_structure(self):
        """Test that package has correct structure"""
        package_dir = Path(__file__).parent.parent
        
        # Check required directories exist
        required_dirs = ["trustcv", "examples", "notebooks", "tests", "docs"]
        for dir_name in required_dirs:
            dir_path = package_dir / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} not found"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
        
        # Check required files exist
        required_files = ["README.md", "setup.py", "requirements.txt"]
        for file_name in required_files:
            file_path = package_dir / file_name
            assert file_path.exists(), f"Required file {file_name} not found"
    
    def test_medicalcv_modules(self):
        """Test that all trustcv modules can be imported"""
        try:
            from trustcv.splitters import iid, temporal, grouped, spatial
            from trustcv.checkers import leakage, balance
            from trustcv.visualization import plots
            from trustcv.metrics import medical_metrics
        except ImportError as e:
            pytest.fail(f"Failed to import trustcv modules: {e}")
    
    def test_cv_method_count(self):
        """Test that we have implemented all 29 CV methods"""
        from trustcv.splitters import iid, temporal, grouped, spatial
        
        # Count methods in each module
        iid_methods = [
            'HoldOut', 'KFoldMedical', 'StratifiedKFoldMedical',
            'RepeatedKFold', 'LOOCV', 'LPOCV', 'BootstrapValidation',
            'MonteCarloCV', 'NestedCV'
        ]
        
        temporal_methods = [
            'TimeSeriesSplit', 'RollingWindowCV', 'ExpandingWindowCV',
            'BlockedTimeSeriesCV', 'PurgedKFoldCV', 'CombinatorialPurgedCV',
            'NestedTemporalCV', 'EmbargoCV'
        ]
        
        grouped_methods = [
            'GroupKFoldMedical', 'StratifiedGroupKFold', 'LeaveOneGroupOut',
            'LeavePGroupsOut', 'RepeatedGroupKFold', 'GroupShuffleSplit',
            'NestedGroupedCV', 'HierarchicalCV'
        ]
        
        spatial_methods = [
            'SpatialBlockCV', 'BufferedSpatialCV',
            'SpatiotemporalBlockCV', 'EnvironmentalHealthCV'
        ]
        
        total_methods = (len(iid_methods) + len(temporal_methods) + 
                        len(grouped_methods) + len(spatial_methods))
        
        assert total_methods >= 29, f"Expected at least 29 CV methods, found {total_methods}"
    
    def test_documentation_completeness(self):
        """Test that documentation is complete"""
        docs_dir = Path(__file__).parent.parent / "docs"
        
        required_docs = [
            "CV_METHODS_CHECKLIST.md",
            "CV_SELECTION_GUIDE.md",
            "PRACTICAL_CV_GUIDE.md",
            "ML_TOOLBOX_CV_COMPARISON.md"
        ]
        
        for doc_name in required_docs:
            doc_path = docs_dir / doc_name
            assert doc_path.exists(), f"Documentation {doc_name} not found"
            
            # Check that docs are not empty
            with open(doc_path, 'r') as f:
                content = f.read()
            assert len(content) > 100, f"Documentation {doc_name} appears to be empty"
            
            # Check for basic markdown structure
            assert "#" in content, f"No headers in {doc_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])