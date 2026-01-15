"""
Medical dataset loaders and generators

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team

Provides easy access to common medical datasets and synthetic data generation
for demonstration and testing purposes.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.preprocessing import StandardScaler


def load_heart_disease() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load heart disease dataset (synthetic version for demo)

    Returns
    -------
    X : pd.DataFrame
        Features (age, blood pressure, cholesterol, etc.)
    y : pd.Series
        Binary target (0: no disease, 1: disease)
    patient_ids : pd.Series
        Patient identifiers

    Examples
    --------
    >>> X, y, patient_ids = load_heart_disease()
    >>> print(f"Dataset shape: {X.shape}")
    >>> print(f"Disease prevalence: {y.mean():.1%}")
    """
    np.random.seed(42)
    n_samples = 1000
    n_patients = 800  # Some patients have multiple records

    # Generate patient IDs (some repeated)
    patient_ids = np.random.choice(range(n_patients), size=n_samples, replace=True)
    patient_ids = pd.Series([f"P{pid:04d}" for pid in patient_ids], name="patient_id")

    # Generate features
    age = np.random.normal(55, 15, n_samples)
    age = np.clip(age, 18, 90)

    # Correlated features
    bmi = np.random.normal(27 + age / 100, 5, n_samples)
    systolic_bp = np.random.normal(120 + age / 2, 20, n_samples)
    diastolic_bp = systolic_bp * 0.6 + np.random.normal(0, 5, n_samples)
    cholesterol = np.random.normal(200 + age / 3, 40, n_samples)
    hdl = np.random.normal(50 - bmi / 2, 10, n_samples)
    ldl = cholesterol - hdl - 20
    glucose = np.random.normal(90 + bmi / 3, 20, n_samples)

    # Binary features
    smoking = np.random.binomial(1, 0.25, n_samples)
    diabetes = np.random.binomial(1, 0.15 + glucose / 1000, n_samples)
    family_history = np.random.binomial(1, 0.3, n_samples)
    exercise = np.random.binomial(1, 0.4 - bmi / 100, n_samples)

    # Create DataFrame
    X = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "cholesterol": cholesterol,
            "hdl": hdl,
            "ldl": ldl,
            "glucose": glucose,
            "smoking": smoking,
            "diabetes": diabetes,
            "family_history": family_history,
            "exercise": exercise,
        }
    )

    # Generate target based on risk factors
    risk_score = (
        (age - 50) / 20 * 0.3
        + (bmi - 25) / 10 * 0.2
        + (systolic_bp - 120) / 40 * 0.2
        + (cholesterol - 200) / 100 * 0.1
        + smoking * 0.3
        + diabetes * 0.2
        + family_history * 0.2
        - exercise * 0.2
    )

    # Convert to probability and sample
    risk_prob = 1 / (1 + np.exp(-risk_score))
    y = np.random.binomial(1, risk_prob)
    y = pd.Series(y, name="heart_disease")

    return X, y, patient_ids


def load_diabetic_readmission() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load diabetic patient readmission dataset (synthetic)

    Returns
    -------
    X : pd.DataFrame
        Features
    y : pd.Series
        Readmission within 30 days (0: no, 1: yes)
    patient_ids : pd.Series
        Patient identifiers
    admission_dates : pd.Series
        Hospital admission dates

    Examples
    --------
    >>> X, y, patient_ids, dates = load_diabetic_readmission()
    >>> print(f"Readmission rate: {y.mean():.1%}")
    """
    np.random.seed(42)
    n_samples = 1500
    n_patients = 500  # Multiple admissions per patient

    # Generate temporal data
    base_date = pd.Timestamp("2020-01-01")
    admission_dates = pd.date_range(base_date, periods=n_samples, freq="6H")
    admission_dates = pd.Series(admission_dates, name="admission_date")

    # Patient IDs with temporal clustering
    patient_ids = []
    for i in range(n_samples):
        if i > 0 and np.random.random() < 0.3:  # 30% chance of readmission
            patient_ids.append(patient_ids[-1])  # Same patient
        else:
            patient_ids.append(f"P{np.random.randint(0, n_patients):04d}")
    patient_ids = pd.Series(patient_ids, name="patient_id")

    # Generate features
    age = np.random.normal(65, 15, n_samples)
    hba1c = np.random.normal(7.5, 2, n_samples)  # Glycated hemoglobin
    glucose = np.random.normal(140, 40, n_samples)
    num_medications = np.random.poisson(5, n_samples)
    num_procedures = np.random.poisson(2, n_samples)
    time_in_hospital = np.random.exponential(5, n_samples)
    num_lab_procedures = np.random.poisson(20, n_samples)
    num_emergency = np.random.poisson(0.5, n_samples)

    # Comorbidities
    comorbidity_score = np.random.poisson(2, n_samples)

    X = pd.DataFrame(
        {
            "age": age,
            "hba1c": hba1c,
            "glucose": glucose,
            "num_medications": num_medications,
            "num_procedures": num_procedures,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_emergency": num_emergency,
            "comorbidity_score": comorbidity_score,
        }
    )

    # Generate readmission based on risk factors
    readmission_risk = (
        (hba1c - 6) / 4 * 0.3
        + (age - 60) / 20 * 0.1
        + time_in_hospital / 20 * 0.2
        + num_emergency / 3 * 0.3
        + comorbidity_score / 5 * 0.2
    )

    readmission_prob = 1 / (1 + np.exp(-readmission_risk))
    y = np.random.binomial(1, readmission_prob)
    y = pd.Series(y, name="readmitted_30days")

    return X, y, patient_ids, admission_dates


def load_cancer_imaging() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load simulated cancer imaging features dataset

    Returns
    -------
    X : np.ndarray
        Imaging features (texture, intensity, shape metrics)
    y : np.ndarray
        Cancer presence (0: benign, 1: malignant)
    patient_ids : np.ndarray
        Patient identifiers

    Examples
    --------
    >>> X, y, patient_ids = load_cancer_imaging()
    >>> print(f"Feature dimensions: {X.shape[1]}")
    >>> print(f"Cancer prevalence: {y.mean():.1%}")
    """
    np.random.seed(42)

    # Generate using sklearn with medical-like characteristics
    X, y = make_classification(
        n_samples=800,
        n_features=50,  # Imaging features
        n_informative=15,  # Relevant features
        n_redundant=10,  # Correlated features
        n_clusters_per_class=3,  # Different cancer subtypes
        weights=[0.7, 0.3],  # 30% cancer prevalence
        flip_y=0.05,  # 5% label noise
        random_state=42,
    )

    # Add feature names that resemble imaging metrics
    feature_names = [
        "mean_intensity",
        "std_intensity",
        "skewness",
        "kurtosis",
        "energy",
        "entropy",
        "contrast",
        "homogeneity",
        "correlation",
        "dissimilarity",
        "asm",
        "max_probability",
    ]

    # Extend feature names
    feature_names.extend([f"texture_feature_{i}" for i in range(len(feature_names), 50)])

    # Generate patient IDs
    patient_ids = np.array([f"IMG_{i:04d}" for i in range(len(X))])

    return X, y, patient_ids


def generate_synthetic_ehr(
    n_samples: int = 1000,
    n_features: int = 20,
    n_patients: Optional[int] = None,
    temporal: bool = False,
    prevalence: float = 0.3,
    random_state: int = 42,
) -> Dict:
    """
    Generate synthetic Electronic Health Record data

    Parameters
    ----------
    n_samples : int
        Number of records
    n_features : int
        Number of features
    n_patients : int, optional
        Number of unique patients (for grouped data)
    temporal : bool
        Whether to include temporal information
    prevalence : float
        Disease prevalence
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary containing X, y, patient_ids, and optionally timestamps

    Examples
    --------
    >>> data = generate_synthetic_ehr(n_samples=1000, temporal=True)
    >>> X, y = data['X'], data['y']
    >>> print(f"Generated {len(X)} records with {X.shape[1]} features")
    """
    np.random.seed(random_state)

    if n_patients is None:
        n_patients = int(n_samples * 0.8)  # Some repeated patients

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=3,
        weights=[1 - prevalence, prevalence],
        flip_y=0.05,
        random_state=random_state,
    )

    # Create feature names
    vital_signs = ["heart_rate", "blood_pressure", "temperature", "respiratory_rate"]
    lab_values = ["wbc", "rbc", "hemoglobin", "platelets", "creatinine", "bun"]

    feature_names = vital_signs + lab_values
    feature_names.extend([f"clinical_feature_{i}" for i in range(len(feature_names), n_features)])

    X = pd.DataFrame(X, columns=feature_names[:n_features])
    y = pd.Series(y, name="outcome")

    # Generate patient IDs with realistic distribution
    patient_distribution = np.random.exponential(1, n_patients)
    patient_distribution = patient_distribution / patient_distribution.sum()
    patient_ids = np.random.choice(range(n_patients), size=n_samples, p=patient_distribution)
    patient_ids = pd.Series([f"EHR_{pid:05d}" for pid in patient_ids], name="patient_id")

    result = {"X": X, "y": y, "patient_ids": patient_ids}

    # Add temporal information if requested
    if temporal:
        base_date = pd.Timestamp("2019-01-01")
        # Exponential inter-arrival times
        inter_arrival = np.random.exponential(1, n_samples)
        timestamps = pd.Series(
            pd.to_datetime(base_date) + pd.to_timedelta(np.cumsum(inter_arrival), unit="D"),
            name="timestamp",
        )
        result["timestamps"] = timestamps

    return result


def generate_temporal_patient_data(
    n_patients: int = 100,
    n_timepoints: int = 12,
    n_features: int = 10,
    outcome_type: str = "binary",
    missing_rate: float = 0.1,
    random_state: int = 42,
) -> Dict:
    """
    Generate temporal patient data with realistic patterns

    Parameters
    ----------
    n_patients : int
        Number of patients
    n_timepoints : int
        Number of time points per patient
    n_features : int
        Number of features
    outcome_type : str
        'binary', 'continuous', or 'survival'
    missing_rate : float
        Proportion of missing values
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary with temporal patient data

    Examples
    --------
    >>> data = generate_temporal_patient_data(n_patients=50, n_timepoints=24)
    >>> print(f"Shape: {data['X'].shape}")
    >>> print(f"Unique patients: {data['patient_ids'].nunique()}")
    """
    np.random.seed(random_state)

    all_data = []

    for patient_id in range(n_patients):
        # Patient-specific baseline
        baseline = np.random.randn(n_features)

        # Generate trajectory
        for t in range(n_timepoints):
            # Add temporal trend and noise
            features = (
                baseline + 0.1 * t * np.random.randn(n_features) + np.random.randn(n_features) * 0.5
            )

            # Add missing values
            if np.random.random() < missing_rate:
                mask = np.random.random(n_features) < 0.3
                features[mask] = np.nan

            record = {
                "patient_id": f"PT_{patient_id:04d}",
                "timepoint": t,
                "days_from_baseline": t * 30,  # Monthly visits
            }

            for f in range(n_features):
                record[f"feature_{f}"] = features[f]

            all_data.append(record)

    df = pd.DataFrame(all_data)

    # Generate outcome based on trajectory
    if outcome_type == "binary":
        # Disease progression
        risk_scores = df.groupby("patient_id")["feature_0"].mean()
        y = (risk_scores > risk_scores.median()).astype(int)
    elif outcome_type == "continuous":
        # Final measurement
        y = df.groupby("patient_id")["feature_0"].last() + np.random.randn(n_patients)
    elif outcome_type == "survival":
        # Time to event
        y = np.random.exponential(100, n_patients)
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")

    return {"X": df, "y": y, "patient_ids": df["patient_id"], "timepoints": df["timepoint"]}
