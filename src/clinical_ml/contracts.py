"""Typed clinical schemas, DTOs, and service protocols."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


def _finite(
    value: float,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be finite.")
    if minimum is not None and num < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    if maximum is not None and num > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}.")
    return num


@dataclass(frozen=True)
class PatientBiometricInput:
    """Validated clinical biometric and medical history request DTO."""

    patient_id: str
    age: int
    sex: str
    weight_kg: float
    height_cm: float
    smoker: str = "Non"
    alcohol: str = "Non"
    activity_level: str = "Moyenne"
    medical_history: str = "Aucun"
    allergy_type: str = "Aucune"

    def __post_init__(self) -> None:
        if not isinstance(self.patient_id, str) or not self.patient_id.strip():
            raise ValueError("patient_id must be a non-empty string.")
        object.__setattr__(self, "patient_id", self.patient_id.strip())
        if isinstance(self.age, bool) or not isinstance(self.age, int) or self.age < 1 or self.age > 120:
            raise ValueError("age must be an integer between 1 and 120.")
        if self.sex not in {"M", "F"}:
            raise ValueError("sex must be 'M' or 'F'.")
        _finite(self.weight_kg, "weight_kg", minimum=20.0, maximum=250.0)
        _finite(self.height_cm, "height_cm", minimum=80.0, maximum=240.0)
        if self.smoker not in {"Non", "Oui"}:
            raise ValueError("smoker must be 'Non' or 'Oui'.")
        if self.alcohol not in {"Non", "Oui"}:
            raise ValueError("alcohol must be 'Non' or 'Oui'.")
        if self.activity_level not in {"Faible", "Moyenne", "Élevée"}:
            raise ValueError("Unsupported activity_level.")
        if self.medical_history not in {"Aucun", "Cardiovasculaire", "Chronique", "Autre"}:
            raise ValueError("Unsupported medical_history.")
        if self.allergy_type not in {"Aucune", "Médicamenteuse", "Autre"}:
            raise ValueError("Unsupported allergy_type.")

    @property
    def bmi(self) -> float:
        meters = self.height_cm / 100.0
        return round(self.weight_kg / (meters * meters), 2)


@dataclass(frozen=True)
class FeatureVector:
    """Normalized clinical numeric vector ready for model inference."""

    patient_id: str
    bmi: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class CompatibilityPrediction:
    """Clinical compatibility decision with confidence band and telemetry."""

    patient_id: str
    bmi: float
    compatible: bool
    compatibility_probability: float
    risk_band: str
    latency_ms: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchEvaluationReport:
    """Percentile compute telemetry report from synthetic cohort scoring."""

    cohort_size: int
    compatible_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_sec: float


class ClinicalFeatureEncoderProtocol(Protocol):
    """Protocol transforming validated biometrics into normalized features."""

    def encode(self, patient: PatientBiometricInput) -> FeatureVector:
        ...


class TreatmentClassifierProtocol(Protocol):
    """Protocol scoring protocol compatibility from normalized vectors."""

    def predict_probability(self, features: FeatureVector) -> float:
        ...


class TelemetrySink(Protocol):
    """Observability interface recording clinical inference events."""

    def record_prediction(self, patient_id: str, probability: float, latency_ms: float) -> None:
        ...
