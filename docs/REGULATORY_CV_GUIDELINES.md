# Regulatory Cross-Validation Guidelines for Medical AI

## FDA, CE MDR & TRIPOD+AI Requirements

*Last updated: January 2025*

> **Disclaimer**: This document provides guidance on cross-validation practices that can support regulatory submissions. TrustCV provides documentation templates and structured outputs that can assist with regulatory documentation, but regulatory compliance depends on the complete device lifecycle and cannot be guaranteed by any single tool. Always consult with regulatory affairs professionals and refer to the official guidance documents for your specific submission requirements.

---

## Table of Contents

1. [Overview & Key Principles](#overview--key-principles)
2. [FDA Requirements (USA)](#fda-requirements-usa)
3. [CE MDR & MDCG Guidelines (Europe)](#ce-mdr--mdcg-guidelines-europe)
4. [TRIPOD+AI Scientific Standards](#tripodai-scientific-standards)
5. [Comparative Requirements](#comparative-requirements)
6. [Implementation Guide](#implementation-guide)
7. [Documentation Checklists](#documentation-checklists)
8. [Official Sources & References](#official-sources--references)

---

## Overview & Key Principles

### Universal Principles for Medical AI Validation

All regulatory frameworks require:

1. **Patient-Level Data Integrity**: All samples from one patient must stay together
2. **Independent Test Set**: Final validation on completely held-out data (15-20%)
3. **Pre-specification**: Validation methods defined before seeing data
4. **Transparency**: Complete documentation of all decisions
5. **Reproducibility**: Random seeds, version control, environment specifications

### Three-Level Validation Architecture

```
Level 1: Development (60-70% of data)
  └── Training Set: Model fitting
  
Level 2: Validation (15-20% of data)  
  └── Cross-Validation: Hyperparameter tuning & method selection
  
Level 3: Testing (15-20% of data)
  └── Held-Out Test Set: Final unbiased performance estimate
```

---

## FDA Requirements (USA)

### Primary Sources

- [FDA AI/ML-Enabled Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [AI/ML Discussion Paper (2019)](https://www.fda.gov/files/medical%20devices/published/US-FDA-Artificial-Intelligence-and-Machine-Learning-Discussion-Paper.pdf)
- [AI/ML Action Plan (2021)](https://www.fda.gov/media/145022/download)
- [Clinical Decision Support Software Guidance (2022)](https://www.fda.gov/media/153486/download)
- [Good Machine Learning Practice (GMLP) Guiding Principles (2021)](https://www.fda.gov/media/122535/download)

### Good Machine Learning Practice (GMLP) - 10 Principles

1. **Multi-disciplinary expertise** throughout lifecycle
2. **Good software engineering** and security practices
3. **Clinical study participants and data sets are representative** of intended patient population
4. **Training/tuning/test sets are independent** and selected appropriately
5. **Reference datasets are based on best available methods**
6. **Model design is tailored to available data** and reflects intended use
7. **Focus on performance of human-AI team**
8. **Testing demonstrates device performance** during clinically relevant conditions
9. **Users provided clear information** on benefits, risks, and limitations
10. **Deployed models monitored** for performance and re-training managed

### FDA Cross-Validation Requirements

#### Development Phase
- **Data Partitioning**:
  - Training: 60-70%
  - Validation/Tuning: 15-20%
  - Test (locked): 15-20%
- **Patient Grouping**: Mandatory for multi-sample patient data
- **Site Diversity**: If multi-site, must validate across sites
- **Temporal Validation**: For time-dependent conditions
- **Demographic Analysis**: Performance across age, sex, race/ethnicity

#### 510(k) Submission Requirements

**Section 13: Software Validation & Verification**
- Complete description of validation methodology
- Justification for data split ratios
- Patient demographics table for each partition
- Performance metrics with 95% confidence intervals
- Subgroup analysis results
- Failure mode analysis
- Comparison to predicate device (if applicable)

#### Common FDA Rejection Reasons

⚠️ **Submissions will be rejected if:**
- Test set contamination (used for ANY decision during development)
- No pre-specified analysis plan
- Missing confidence intervals
- Inadequate sample size justification
- No subgroup analysis for protected classes
- Patient data appears in multiple partitions

#### Predetermined Change Control Plan (PCCP)

For AI/ML devices that will be updated post-market:
- **Retraining Protocol**: How will model be retrained?
- **Performance Monitoring**: Metrics and thresholds
- **Update Validation**: Cross-validation strategy for updates
- **Data Drift Detection**: Methods to identify distribution shifts
- **Version Control**: Track which validation for which version

### FDA-Compliant Code Example

```python
from sklearn.model_selection import train_test_split

# First: Separate test set (LOCK THIS AWAY)
X_dev, X_test, y_dev, y_test, patients_dev, patients_test = train_test_split(
    X, y, patient_ids,
    test_size=0.20,  # 20% for final test
    stratify=y,       # Maintain class balance
    random_state=42   # Pre-specified seed
)

# Second: Split development into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.20,   # 20% of 80% = 16% total for validation
    stratify=y_dev,
    random_state=42
)

# Use validation for hyperparameter tuning
# NEVER touch X_test until final evaluation
```

---

## CE MDR & MDCG Guidelines (Europe)

### Primary Sources

- [MDR 2017/745 Full Text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32017R0745)
- [MDCG 2021-24: Classification of Medical Device Software](https://health.ec.europa.eu/system/files/2021-10/md_mdcg_2021_24_en_0.pdf)
- [MDCG 2020-1: Clinical Evaluation of Medical Device Software](https://health.ec.europa.eu/system/files/2020-09/md_mdcg_2020_1_guidance_clinic_eva_md_software_en_0.pdf)
- [MDCG 2019-11: Software Qualification and Classification](https://health.ec.europa.eu/system/files/2019-10/md_mdcg_2019_11_guidance_qualification_classification_software_en_0.pdf)
- [MDCG 2019-16: Cybersecurity Guidance](https://health.ec.europa.eu/system/files/2019-10/md_mdcg_2019_16_guidance_cybersecurity_en_0.pdf)

### MDR Annex I - Essential Requirements

**Chapter II, Section 17.2: Software validation requirements**
- Software shall be validated according to state of the art
- Validation shall demonstrate conformity with intended purpose
- Lifecycle approach including verification and validation
- Clinical evidence required for Class IIa and higher

### MDCG 2021-24: AI/ML Classification

| Class | Description | Validation Requirements |
|-------|-------------|------------------------|
| **Class I** | Low risk, no direct patient impact | Basic validation |
| **Class IIa** | Inform clinical decisions | Clinical evaluation required |
| **Class IIb** | Direct diagnosis/monitoring vital functions | Extensive clinical validation |
| **Class III** | Life-supporting/life-saving | Highest level validation + clinical trials |

### MDCG 2020-1: Clinical Evaluation Process

1. **Stage 0 - Scope**: Define clinical claims and intended use
2. **Stage 1 - Data**: Identify pertinent data
   - Clinical data from device under evaluation
   - Clinical data from equivalent device
   - Relevant scientific literature
3. **Stage 2 - Appraisal**: Evaluate data quality and relevance
4. **Stage 3 - Analysis**: Demonstrate:
   - Performance (analytical validation)
   - Clinical performance (clinical validation)
   - Clinical utility
5. **Stage 4 - Report**: Clinical Evaluation Report (CER)

### Key Differences from FDA

| Aspect | CE MDR | FDA |
|--------|--------|-----|
| Classification System | I, IIa, IIb, III (risk-based) | Class I, II, III + De Novo |
| Clinical Evidence | Clinical Evaluation Report (CER) | Clinical validation in 510(k) |
| Post-Market | PMCF (Post-Market Clinical Follow-up) | Post-market surveillance + PCCP |
| AI/ML Updates | Significant change requires new assessment | PCCP allows predetermined changes |
| Notified Body | Required for Class IIa and above | FDA direct review |

---

## TRIPOD+AI Scientific Standards

### Primary Sources

- [TRIPOD+AI Statement - EQUATOR Network](https://www.equator-network.org/reporting-guidelines/tripod-statement/)
- [TRIPOD+AI Statement Publication (BMJ 2024)](https://www.bmj.com/content/385/bmj-2023-078378)
- [Official TRIPOD Statement Website](https://www.tripod-statement.org/)
- [CONSORT-AI and SPIRIT-AI Guidelines](https://www.nature.com/articles/s41591-020-1034-x)

### TRIPOD+AI Checklist Items for Cross-Validation

- **Item 10b - Data Partitioning**: Describe how data was split (random, temporal, geographic)
- **Item 10c - Cross-validation**: Report exact CV method, number of folds, stratification
- **Item 13b - Patient Clustering**: How multiple measurements per patient were handled
- **Item 14b - Missing Data**: How missing data was handled in each fold
- **Item 15b - Model Specification**: Report if model selection used same data as evaluation
- **Item 16 - Performance Metrics**: Report with confidence intervals for each fold
- **Item 18b - Model Updating**: Describe any calibration or updating procedures

### Best Practices from TRIPOD+AI

1. **Transparent Reporting**:
   - Report performance for EACH fold, not just mean
   - Include distribution plots of fold performances
   - Document any folds that failed or had issues

2. **Data Dependencies**:
   - Clearly state clustering (patient, site, time)
   - Justify independence assumptions
   - Report effective sample size after clustering

3. **Reproducibility**:
   - Provide all random seeds
   - Document software versions
   - Share code when possible

4. **Uncertainty Quantification**:
   - Bootstrap confidence intervals
   - Calibration plots for each fold
   - Decision curve analysis

### TRIPOD+AI Compliant Code Example

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def tripod_compliant_cv(X, y, patient_ids, model):
    """
    TRIPOD+AI compliant cross-validation with full reporting
    """
    # Item 10c: Specify exact CV method
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results for each fold (Item 16)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Item 13b: Ensure patient-level splitting
        train_patients = patient_ids[train_idx]
        val_patients = patient_ids[val_idx]
        assert len(np.intersect1d(train_patients, val_patients)) == 0
        
        # Train model
        model.fit(X[train_idx], y[train_idx])
        
        # Predictions
        y_pred_proba = model.predict_proba(X[val_idx])[:, 1]
        
        # Calculate metrics for this fold
        fold_auc = roc_auc_score(y[val_idx], y_pred_proba)
        
        # Store detailed results
        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_patients_train': len(np.unique(train_patients)),
            'n_patients_val': len(np.unique(val_patients)),
            'auc': fold_auc,
            'y_true': y[val_idx],
            'y_pred': y_pred_proba
        })
        
        print(f"Fold {fold_idx + 1}: AUC = {fold_auc:.3f}")
    
    # Item 16: Report aggregate metrics with CI
    aucs = [r['auc'] for r in fold_results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    
    print(f"\nOverall Performance:")
    print(f"Mean AUC: {mean_auc:.3f} (SD: {std_auc:.3f})")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Individual fold AUCs: {aucs}")
    
    return fold_results
```

---

## Comparative Requirements

| Requirement | FDA | CE MDR/MDCG | TRIPOD+AI |
|------------|-----|-------------|-----------|
| **Test Set** | ✅ Mandatory locked (15-20%) | ✅ Required for Class IIa+ | ✅ Strongly recommended |
| **Patient Grouping** | ✅ Mandatory | ✅ Required when applicable | ✅ Must report clustering |
| **CV Method** | Pre-specified, justified | State of the art | Fully documented with code |
| **Confidence Intervals** | ✅ Required (95% CI) | ✅ Required | ✅ Required with method |
| **Subgroup Analysis** | ✅ Mandatory (age, sex, race) | Risk-based requirement | ✅ Recommended |
| **External Validation** | Often for De Novo | Depends on class | ✅ Strongly recommended |
| **Performance per Fold** | Summary statistics ok | Summary statistics ok | ✅ Must report each fold |
| **Code Sharing** | Not required | Not required | ✅ Strongly encouraged |
| **Update Strategy** | PCCP required | New assessment | Document update methods |
| **Clinical Evidence** | Clinical validation study | Clinical Evaluation Report | Clinical utility analysis |

---

## Implementation Guide

### Complete Regulatory-Compliant Validation Pipeline

```python
"""
Regulatory-Compliant Cross-Validation Pipeline
Meets FDA, CE MDR, and TRIPOD+AI requirements
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import json
from datetime import datetime
import hashlib

class RegulatoryCompliantValidator:
    """
    Implements validation meeting FDA, CE MDR, and TRIPOD+AI standards
    """
    
    def __init__(self, regulatory_standard='all', device_class='IIa'):
        self.regulatory_standard = regulatory_standard
        self.device_class = device_class
        self.validation_plan = self._create_validation_plan()
        self.audit_trail = []
        
    def _create_validation_plan(self):
        """Pre-specify validation plan (FDA requirement)"""
        plan = {
            'created_at': datetime.now().isoformat(),
            'test_set_ratio': 0.20,
            'validation_ratio': 0.20,
            'cv_method': 'stratified_patient_kfold',
            'n_folds': 5,
            'random_seed': 42,
            'stratification': 'outcome',
            'patient_grouping': True,
            'confidence_level': 0.95,
            'subgroups': ['age_group', 'sex', 'race', 'site'],
            'performance_metrics': [
                'sensitivity', 'specificity', 'ppv', 'npv',
                'accuracy', 'auc_roc', 'auc_pr', 'calibration'
            ]
        }
        
        # Lock the plan with hash (FDA: pre-specification)
        plan['hash'] = hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest()
        
        return plan
    
    def split_data(self, X, y, patient_ids, demographics=None):
        """
        Three-way split meeting all regulatory requirements
        """
        # FDA: Ensure patient-level splitting
        unique_patients = np.unique(patient_ids)
        patient_labels = pd.DataFrame({
            'patient_id': unique_patients,
            'label': [y[patient_ids == pid][0] for pid in unique_patients]
        })
        
        # First split: Separate test set (FDA: locked test set)
        patients_dev, patients_test = train_test_split(
            patient_labels['patient_id'],
            test_size=self.validation_plan['test_set_ratio'],
            stratify=patient_labels['label'],
            random_state=self.validation_plan['random_seed']
        )
        
        # Create masks
        dev_mask = np.isin(patient_ids, patients_dev)
        test_mask = np.isin(patient_ids, patients_test)
        
        # Split data
        X_dev, X_test = X[dev_mask], X[test_mask]
        y_dev, y_test = y[dev_mask], y[test_mask]
        patient_ids_dev = patient_ids[dev_mask]
        patient_ids_test = patient_ids[test_mask]
        
        # Lock test set (FDA requirement)
        self.locked_test_set = {
            'X': X_test.copy(),
            'y': y_test.copy(),
            'patient_ids': patient_ids_test.copy(),
            'locked_at': datetime.now().isoformat(),
            'hash': hashlib.sha256(X_test.tobytes() + y_test.tobytes()).hexdigest()
        }
        
        return X_dev, y_dev, patient_ids_dev
```

---

## Documentation Checklists

These checklists can help ensure your validation documentation addresses common regulatory requirements. They are not exhaustive - always refer to official guidance for your specific submission.

### FDA 510(k) Documentation Checklist

- [ ] Pre-specified validation plan documented before data analysis
- [ ] Test set (15-20%) locked and never used during development
- [ ] Patient-level data splitting (no patient in multiple sets)
- [ ] 95% confidence intervals for all performance metrics
- [ ] Subgroup analysis by age, sex, race/ethnicity
- [ ] Failure mode analysis documented
- [ ] Software version control and reproducibility documented
- [ ] PCCP included if continuous learning planned

### CE MDR Technical Documentation Checklist

- [ ] Device classification determined (I, IIa, IIb, or III)
- [ ] Clinical Evaluation Report (CER) prepared
- [ ] State-of-the-art validation methodology used
- [ ] Risk management per ISO 14971
- [ ] Usability engineering per IEC 62366
- [ ] Post-Market Clinical Follow-up (PMCF) plan
- [ ] Notified Body selected (for Class IIa+)

### TRIPOD+AI Reporting Checklist

- [ ] Cross-validation method fully specified
- [ ] Performance reported for each fold individually
- [ ] Patient clustering handled and reported
- [ ] Missing data handling documented
- [ ] Calibration plots included
- [ ] Code and random seeds provided
- [ ] Limitations clearly stated

---

## Official Sources & References

### FDA Resources
- [FDA AI/ML Medical Devices Main Page](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)
- [AI/ML-Based SaMD Action Plan (2021)](https://www.fda.gov/media/145022/download)
- [Good Machine Learning Practice Guiding Principles](https://www.fda.gov/media/122535/download)
- [FDA Guidance Documents Database](https://www.fda.gov/regulatory-information/search-fda-guidance-documents)

### CE MDR & MDCG Resources
- [MDR 2017/745 Full Legal Text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32017R0745)
- [MDCG Guidance Documents](https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en)
- [MDCG 2021-24: Classification of Medical Device Software](https://health.ec.europa.eu/system/files/2021-10/md_mdcg_2021_24_en_0.pdf)
- [MDCG 2020-1: Clinical Evaluation Guidance](https://health.ec.europa.eu/system/files/2020-09/md_mdcg_2020_1_guidance_clinic_eva_md_software_en_0.pdf)

### TRIPOD+AI Resources
- [TRIPOD+AI on EQUATOR Network](https://www.equator-network.org/reporting-guidelines/tripod-statement/)
- [TRIPOD+AI Statement (BMJ 2024)](https://www.bmj.com/content/385/bmj-2023-078378)
- [Official TRIPOD Website](https://www.tripod-statement.org/)
- [CONSORT-AI and SPIRIT-AI Extensions](https://www.nature.com/articles/s41591-020-1034-x)

### Additional Standards
- [ISO 13485:2016 - Medical Device Quality Management](https://www.iso.org/standard/59752.html)
- [ISO 14971:2019 - Risk Management](https://www.iso.org/standard/72704.html)
- [IEC 62304:2006+A1:2015 - Medical Device Software](https://webstore.iec.ch/publication/21170)
- [IEC 62366-1:2015 - Usability Engineering](https://webstore.iec.ch/publication/21863)

---

## Disclaimer

This guide provides educational information based on publicly available regulatory documents. Always consult with regulatory professionals and refer to the latest official guidelines for your specific submission. Requirements may vary based on device classification, intended use, and target markets.

---

*Last updated: January 2025*  
*Part of the trustcv toolkit - https://github.com/ki-smile/trustcv*