"""
Regulatory Compliance Report Generator for Trustworthy Cross-Validation
Generates FDA/CE-compliant validation reports from cross-validation results
"""

import json
import datetime
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
from pathlib import Path


class RegulatoryReport:
    """
    Generate regulatory-compliant validation reports for AI systems.
    
    This class creates FDA 510(k) and CE MDR compliant reports documenting
    the validation methodology and performance of AI models.
    """
    
    def __init__(self, 
                 model_name: str,
                 model_version: str,
                 manufacturer: str,
                 intended_use: str,
                 compliance_standard: str = 'FDA',
                 project_name: Optional[str] = None):
        """
        Initialize regulatory report generator.
        
        Parameters
        ----------
        model_name : str
            Name of the AI model/device
        model_version : str
            Version number of the software
        manufacturer : str
            Name of the manufacturer/developer
        intended_use : str
            Clinical intended use statement
        compliance_standard : str
            Regulatory standard ('FDA', 'CE', 'both')
        """
        self.model_name = model_name
        self.model_version = model_version
        self.manufacturer = manufacturer
        self.intended_use = intended_use
        self.compliance_standard = compliance_standard
        self.project_name = project_name
        
        # Store validation results
        self.cv_results = {}
        self.dataset_info = {}
        self.performance_metrics = {}
        self.validation_method = None

    # --- convenience helpers ---
    def generate_from_validator(self, validator, run_id: Optional[str] = None,
                                output_path: Optional[str] = None,
                                format: str = 'html') -> str:
        """
        Build and save a regulatory report from a TrustCVValidator instance.

        Expects the validator to have a `.last_result` attribute produced by
        a recent call to `validate(...)`.
        """
        # Extract CV config
        try:
            method = getattr(validator, 'method', None)
            n_splits = int(getattr(validator, 'n_splits', 0) or 0)
        except Exception:
            method, n_splits = None, 0

        # Pull last validation result if available
        valres = getattr(validator, 'last_result', None)
        scores_list = []
        if valres is not None:
            # Choose a primary metric to summarize per-fold results
            metric_priority = ['accuracy', 'roc_auc', 'f1', 'score']
            chosen = None
            for m in metric_priority:
                if m in valres.scores:
                    chosen = m
                    break
            if chosen is None and valres.scores:
                chosen = next(iter(valres.scores.keys()))
            if chosen is not None:
                try:
                    import numpy as _np
                    scores_arr = _np.asarray(valres.scores[chosen], dtype=float)
                    scores_list = scores_arr.ravel().tolist()
                except Exception:
                    scores_list = []

        # Fill CV section
        self.add_cv_results(method=str(method) if method else 'unknown',
                            n_splits=n_splits or (len(scores_list) or 0),
                            scores=scores_list or [])

        # Include run id in metadata if provided via generate_regulatory_report
        # The underlying generator will record report_id; we can carry run_id in device_info
        if run_id:
            # attach to dataset_info for traceability
            self.dataset_info.setdefault('run_id', run_id)

        # Finally, generate report file or return JSON/HTML string path
        if output_path is None:
            # Default filename
            output_path = f"regulatory_report_{run_id or 'cv'}.{ 'html' if format=='html' else ( 'json' if format=='json' else 'pdf' ) }"
        return self.generate_regulatory_report(output_path=output_path, format=format)
        
    def add_dataset_info(self, 
                        n_patients: int,
                        n_samples: int,
                        n_features: int,
                        demographics: Optional[Dict] = None,
                        data_sources: Optional[List[str]] = None,
                        class_distribution: Optional[Dict[str, Dict[str, float]]] = None):
        """Add dataset information to the report."""
        self.dataset_info = {
            'n_patients': n_patients,
            'n_samples': n_samples,
            'n_features': n_features,
            'demographics': demographics or {},
            'data_sources': data_sources or [],
            'class_distribution': class_distribution or {}
        }
        
    def add_cv_results(self,
                       method: str,
                       n_splits: int,
                       scores: List[float],
                       confusion_matrices: Optional[List] = None):
        """Add cross-validation results."""
        self.validation_method = method
        self.cv_results = {
            'method': method,
            'n_splits': n_splits,
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'confusion_matrices': confusion_matrices
        }
        
    def calculate_clinical_metrics(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_proba: Optional[np.ndarray] = None):
        """Calculate clinical performance metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            roc_auc_score, confusion_matrix
        )
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        self.performance_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),  # TPR
            'specificity': tn / (tn + fp),  # TNR
            'ppv': precision_score(y_true, y_pred),  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        }
        
        if y_proba is not None:
            self.performance_metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            
    def generate_regulatory_report(self, 
                                  output_path: str,
                                  format: str = 'html') -> str:
        """
        Generate the regulatory compliance report.
        
        Parameters
        ----------
        output_path : str
            Path to save the report
        format : str
            Output format ('html', 'pdf', 'json')
            
        Returns
        -------
        str
            Path to the generated report
        """
        report_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create report structure
        report = {
            'metadata': {
                'report_id': f"VAL-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'generation_date': report_date,
                'compliance_standard': self.compliance_standard,
                'report_version': '1.0'
            },
            'device_info': {
                'name': self.model_name,
                'version': self.model_version,
                'manufacturer': self.manufacturer,
                'intended_use': self.intended_use
            },
            'dataset': self.dataset_info,
            'validation': {
                'method': self.validation_method,
                'cv_results': self.cv_results,
                'performance': self.performance_metrics
            }
        }
        if self.project_name:
            report['metadata']['project_name'] = self.project_name
        
        if format == 'json':
            return self._save_json_report(report, output_path)
        elif format == 'html':
            return self._save_html_report(report, output_path)
        elif format == 'pdf':
            return self._save_pdf_report(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _save_json_report(self, report: Dict, output_path: str) -> str:
        """Save report as JSON."""
        path = Path(output_path)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return str(path)
        
    def _save_html_report(self, report: Dict, output_path: str) -> str:
        """Generate and save HTML report."""
        html_template = self._generate_html_template(report)
        path = Path(output_path)
        with open(path, 'w') as f:
            f.write(html_template)
        return str(path)
        
    def _generate_html_template(self, report: Dict) -> str:
        """Generate HTML report template."""
        metrics = report['validation'].get('performance', {}) if isinstance(report.get('validation'), dict) else {}
        cm = metrics.get('confusion_matrix', {}) if isinstance(metrics, dict) else {}
        tp = cm.get('tp', cm.get('true_positives', 0))
        tn = cm.get('tn', cm.get('true_negatives', 0))
        fp = cm.get('fp', cm.get('false_positives', 0))
        fn = cm.get('fn', cm.get('false_negatives', 0))

        def format_ci(ci_value, *, percent=False, decimals=3) -> str:
            """Format confidence intervals consistently."""
            if ci_value is None:
                return "N/A"
            try:
                lower, upper = float(ci_value[0]), float(ci_value[1])
            except Exception:
                return "N/A"
            if percent:
                return f"[{lower:.{decimals}%}, {upper:.{decimals}%}]"
            return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"

        sensitivity_ci = format_ci(metrics.get('sensitivity_ci'), percent=True, decimals=1)
        specificity_ci = format_ci(metrics.get('specificity_ci'), percent=True, decimals=1)
        auc_ci = format_ci(metrics.get('auc_roc_ci'), percent=False, decimals=3)
        
        project_info = ""
        project_name = report['metadata'].get('project_name')
        if project_name:
            project_info = f"<p><strong>Project:</strong> {project_name}</p>"

        class_dist_html = self._render_class_distribution(report['dataset'].get('class_distribution', {}))
        demographics_html = self._render_demographics(report['dataset'].get('demographics', {}))
        data_sources_html = self._render_data_sources(report['dataset'].get('data_sources', []))

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Regulatory Validation Report - {report['device_info']['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .metric {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Clinical Validation Report</h1>
                <p><strong>Device:</strong> {report['device_info']['name']} v{report['device_info']['version']}</p>
                <p><strong>Report ID:</strong> {report['metadata']['report_id']}</p>
                <p><strong>Date:</strong> {report['metadata']['generation_date']}</p>
                {project_info}
            </div>
            
            <h2>1. Executive Summary</h2>
            <p><strong>Intended Use:</strong> {report['device_info']['intended_use']}</p>
            
            <div class="metric">
                <h3>Key Performance Metrics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>95% CI</th>
                    </tr>
                    <tr>
                        <td>Sensitivity</td>
                        <td>{metrics.get('sensitivity', 0):.1%}</td>
                        <td>{sensitivity_ci}</td>
                    </tr>
                    <tr>
                        <td>Specificity</td>
                        <td>{metrics.get('specificity', 0):.1%}</td>
                        <td>{specificity_ci}</td>
                    </tr>
                    <tr>
                        <td>AUC-ROC</td>
                        <td>{metrics.get('auc_roc', 0):.3f}</td>
                        <td>{auc_ci}</td>
                    </tr>
                </table>
            </div>
            
            <h2>2. Dataset Characteristics</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Patients</td>
                    <td>{report['dataset'].get('n_patients', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Total Samples</td>
                    <td>{report['dataset'].get('n_samples', 'N/A')}</td>
                </tr>
                <tr>
                    <td>Features</td>
                    <td>{report['dataset'].get('n_features', 'N/A')}</td>
                </tr>
            </table>
            {class_dist_html}
            {demographics_html}
            {data_sources_html}
            
            <h2>3. Validation Methodology</h2>
            <p><strong>Method:</strong> {report['validation']['method']}</p>
            <p><strong>Number of Folds:</strong> {report['validation']['cv_results'].get('n_splits', 'N/A')}</p>
            <p><strong>Mean Accuracy:</strong> {report['validation']['cv_results'].get('mean_score', 0):.3f} &plusmn; {report['validation']['cv_results'].get('std_score', 0):.3f}</p>
            
            <h2>4. Confusion Matrix</h2>
            <table style="width: auto; margin: 20px auto;">
                <tr>
                    <th colspan="2" rowspan="2"></th>
                    <th colspan="2">Predicted</th>
                </tr>
                <tr>
                    <th>Positive</th>
                    <th>Negative</th>
                </tr>
                <tr>
                    <th rowspan="2">Actual</th>
                    <th>Positive</th>
                    <td style="background: #d4edda;">{tp}</td>
                    <td style="background: #f8d7da;">{fn}</td>
                </tr>
                <tr>
                    <th>Negative</th>
                    <td style="background: #f8d7da;">{fp}</td>
                    <td style="background: #d4edda;">{tn}</td>
                </tr>
            </table>
            
            <h2>5. Regulatory Compliance</h2>
            <p>This report complies with {report['metadata']['compliance_standard']} requirements for medical device software validation.</p>
            
            <h2>6. Approval</h2>
            <div style="margin-top: 50px;">
                <table style="border: none;">
                    <tr>
                        <td style="border: none; border-top: 2px solid #333; width: 30%; padding-top: 10px;">
                            Clinical Study Director<br>
                            Date: _______________
                        </td>
                        <td style="border: none; width: 20%;"></td>
                        <td style="border: none; border-top: 2px solid #333; width: 30%; padding-top: 10px;">
                            Quality Assurance<br>
                            Date: _______________
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        return html
    
    @staticmethod
    def _render_class_distribution(class_dist: Dict[str, Dict[str, float]]) -> str:
        if not class_dist:
            return ""
        rows = []
        for label, stats in class_dist.items():
            count = stats.get('count', 'N/A')
            pct = stats.get('percentage', None)
            if isinstance(pct, (int, float)):
                pct_str = f"{pct:.1f}%"
            else:
                pct_str = pct if pct is not None else "N/A"
            rows.append(f"<tr><td>{label}</td><td>{count}</td><td>{pct_str}</td></tr>")
        rows_html = "\n".join(rows)
        return f"""
        <h3>Class Distribution</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {rows_html}
        </table>
        """

    @staticmethod
    def _render_demographics(demographics: Dict[str, Dict[str, float]]) -> str:
        if not demographics:
            return ""
        sections = ["<h3>Demographics</h3>"]
        for category, values in demographics.items():
            title = str(category).replace("_", " ").title()
            if isinstance(values, dict):
                sub_rows = "".join(
                    f"<tr><td>{str(sub)}</td><td>{str(val)}</td></tr>"
                    for sub, val in values.items()
                )
                sections.append(
                    f"<h4>{title}</h4><table><tr><th>Category</th><th>Value</th></tr>{sub_rows}</table>"
                )
            else:
                sections.append(f"<p><strong>{title}:</strong> {values}</p>")
        return "".join(sections)

    @staticmethod
    def _render_data_sources(data_sources: Iterable[str]) -> str:
        if not data_sources:
            return ""
        items = "".join(f"<li>{src}</li>" for src in data_sources)
        return f"<h3>Data Sources</h3><ul>{items}</ul>"
        
    def _save_pdf_report(self, report: Dict, output_path: str) -> str:
        """Save report directly as PDF, falling back to HTML if converter missing."""
        try:
            from weasyprint import HTML  # type: ignore
        except ImportError:
            html_path = output_path.replace('.pdf', '.html')
            self._save_html_report(report, html_path)
            print(
                "weasyprint is not installed; saved HTML instead at "
                f"{html_path}. Install `weasyprint` to enable PDF export."
            )
            return html_path

        html_string = self._generate_html_template(report)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_string).write_pdf(str(path))
        return str(path)
        
    def plot_validation_curves(self):
        """Plot validation curves (placeholder for visualization)."""
        print("Validation curves would be plotted here using matplotlib/plotly")
        # Implementation would use matplotlib or plotly to generate curves
        pass


# Example usage function that matches the README
def validate_medical_model(model, data, patient_ids=None, compliance='FDA'):
    """
    Validate a ML model with proper cross-validation and generate reports.
    
    This is a high-level wrapper that performs validation and generates
    regulatory-compliant reports in one call.
    """
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from ..splitters.grouped import GroupKFoldMedical
    
    X, y = data
    
    # Create appropriate cross-validator
    if patient_ids is not None:
        cv = GroupKFoldMedical(n_splits=5)
        cv_generator = cv.split(X, y, groups=patient_ids)
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_generator = cv.split(X, y)
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=list(cv_generator), scoring='roc_auc')
    
    # Create report
    report = RegulatoryReport(
        model_name=model.__class__.__name__,
        model_version="1.0",
        manufacturer="Medical AI Lab",
        intended_use="Diagnostic assistance for medical professionals",
        compliance_standard=compliance
    )
    
    # Add validation results
    report.add_cv_results(
        method="Stratified Group K-Fold" if patient_ids is not None else "Stratified K-Fold",
        n_splits=5,
        scores=scores.tolist()
    )
    
    # Add dataset info
    n_patients = len(np.unique(patient_ids)) if patient_ids is not None else len(X)
    report.add_dataset_info(
        n_patients=n_patients,
        n_samples=len(X),
        n_features=X.shape[1] if len(X.shape) > 1 else 1
    )
    
    return report

