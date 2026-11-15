"""Clean-architecture clinical patient compatibility classification core."""

from src.clinical_ml.contracts import (
    BatchEvaluationReport,
    ClinicalFeatureEncoderProtocol,
    CompatibilityPrediction,
    FeatureVector,
    PatientBiometricInput,
    TelemetrySink,
    TreatmentClassifierProtocol,
)
from src.clinical_ml.service import (
    ClinicalContainer,
    ClinicalFeatureEncoder,
    ClinicalTreatmentService,
    InMemoryTelemetrySink,
    LogisticTreatmentClassifier,
    check_liveness,
    check_readiness,
    get_health_status,
    run_synthetic_cohort_benchmark,
)

__all__ = [
    "BatchEvaluationReport",
    "ClinicalContainer",
    "ClinicalFeatureEncoder",
    "ClinicalFeatureEncoderProtocol",
    "ClinicalTreatmentService",
    "CompatibilityPrediction",
    "FeatureVector",
    "InMemoryTelemetrySink",
    "LogisticTreatmentClassifier",
    "PatientBiometricInput",
    "TelemetrySink",
    "TreatmentClassifierProtocol",
    "check_liveness",
    "check_readiness",
    "get_health_status",
    "run_synthetic_cohort_benchmark",
]
