"""
Medical datasets for demonstration and testing

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

from .loaders import (
    generate_synthetic_ehr,
    generate_temporal_patient_data,
    load_cancer_imaging,
    load_diabetic_readmission,
    load_heart_disease,
)

__all__ = [
    "load_heart_disease",
    "load_diabetic_readmission",
    "load_cancer_imaging",
    "generate_synthetic_ehr",
    "generate_temporal_patient_data",
]
