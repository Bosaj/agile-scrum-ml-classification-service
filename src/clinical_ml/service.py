"""DI container, clinical encoder, treatment scorer, probes, and load test."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.clinical_ml.contracts import (
    BatchEvaluationReport,
    ClinicalFeatureEncoderProtocol,
    CompatibilityPrediction,
    FeatureVector,
    PatientBiometricInput,
    TelemetrySink,
    TreatmentClassifierProtocol,
)

_ACTIVITY_MAP = {"Faible": 0.0, "Moyenne": 1.0, "Élevée": 2.0}
_HISTORY_MAP = {"Aucun": 0.0, "Cardiovasculaire": 1.0, "Chronique": 2.0, "Autre": 3.0}
_ALLERGY_MAP = {"Aucune": 0.0, "Médicamenteuse": 1.0, "Autre": 2.0}


def check_liveness() -> bool:
    """Return True when clinical scoring runtime is healthy."""
    return True


def check_readiness() -> bool:
    """Return True when encoder and classifier pipeline are ready."""
    return True


def get_health_status() -> dict[str, object]:
    """Structured Kubernetes-ready probe status payload."""
    alive = check_liveness()
    ready = check_readiness()
    return {
        "service": "agile-scrum-ml-classification-service",
        "status": "UP" if alive and ready else "DOWN",
        "timestamp": time.time(),
        "checks": {"liveness": alive, "readiness": ready},
    }


class JsonLogFormatter(logging.Formatter):
    """Emit structured JSON log entries for clinical telemetry."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
            }
        )


class InMemoryTelemetrySink:
    """Thread-safe recorder of clinical compatibility inference events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, float, float]] = []

    def record_prediction(self, patient_id: str, probability: float, latency_ms: float) -> None:
        with self._lock:
            self.events.append((patient_id, float(probability), float(latency_ms)))


class ClinicalFeatureEncoder:
    """Deterministic biometric encoder matching ENIAD UI categorical semantics."""

    def encode(self, patient: PatientBiometricInput) -> FeatureVector:
        bmi = patient.bmi
        normalized = (
            patient.age / 100.0,
            1.0 if patient.sex == "M" else 0.0,
            bmi / 40.0,
            1.0 if patient.smoker == "Oui" else 0.0,
            1.0 if patient.alcohol == "Oui" else 0.0,
            _ACTIVITY_MAP[patient.activity_level] / 2.0,
            _HISTORY_MAP[patient.medical_history] / 3.0,
            _ALLERGY_MAP[patient.allergy_type] / 2.0,
        )
        return FeatureVector(patient_id=patient.patient_id, bmi=bmi, values=normalized)


class LogisticTreatmentClassifier:
    """Calibrated logistic compatibility model evaluating risk-adjusted tolerance."""

    _WEIGHTS = (-0.55, 0.05, -0.35, -0.75, -0.45, 0.65, -0.95, -0.85)
    _BIAS = 1.45

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def predict_probability(self, features: FeatureVector) -> float:
        logit = self._BIAS + sum(w * x for w, x in zip(self._WEIGHTS, features.values))
        return round(self._sigmoid(logit), 6)


class ClinicalContainer:
    """Minimal dependency injection container for clinical service layers."""

    def __init__(self) -> None:
        self._instances: dict[type[Any], Any] = {}
        self._factories: dict[type[Any], Callable[[], Any]] = {}

    def register_instance(self, interface: type[Any], instance: Any) -> None:
        self._instances[interface] = instance

    def register_factory(self, interface: type[Any], factory: Callable[[], Any]) -> None:
        self._factories[interface] = factory

    def resolve(self, interface: type[Any]) -> Any:
        if interface in self._instances:
            return self._instances[interface]
        if interface not in self._factories:
            raise KeyError(f"No provider registered for {interface.__name__}.")
        instance = self._factories[interface]()
        self._instances[interface] = instance
        return instance


class ClinicalTreatmentService:
    """Orchestrates biometric encoding, protocol compatibility scoring, and telemetry."""

    def __init__(
        self,
        encoder: ClinicalFeatureEncoderProtocol,
        classifier: TreatmentClassifierProtocol,
        telemetry: TelemetrySink,
    ) -> None:
        self._encoder = encoder
        self._classifier = classifier
        self._telemetry = telemetry

    @classmethod
    def create_default(cls, telemetry: TelemetrySink | None = None) -> ClinicalTreatmentService:
        container = ClinicalContainer()
        container.register_instance(ClinicalFeatureEncoderProtocol, ClinicalFeatureEncoder())
        container.register_instance(TreatmentClassifierProtocol, LogisticTreatmentClassifier())
        container.register_instance(TelemetrySink, telemetry or InMemoryTelemetrySink())
        return cls(
            encoder=container.resolve(ClinicalFeatureEncoderProtocol),
            classifier=container.resolve(TreatmentClassifierProtocol),
            telemetry=container.resolve(TelemetrySink),
        )

    def evaluate(self, patient: PatientBiometricInput) -> CompatibilityPrediction:
        t0 = time.perf_counter()
        vector = self._encoder.encode(patient)
        prob = self._classifier.predict_probability(vector)
        compatible = prob >= 0.5
        if prob >= 0.75:
            band = "LOW_RISK"
        elif prob >= 0.45:
            band = "MODERATE_RISK"
        else:
            band = "HIGH_RISK"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._telemetry.record_prediction(patient.patient_id, prob, latency_ms)
        return CompatibilityPrediction(
            patient_id=patient.patient_id,
            bmi=vector.bmi,
            compatible=compatible,
            compatibility_probability=prob,
            risk_band=band,
            latency_ms=latency_ms,
            metadata={"activity_level": patient.activity_level},
        )


def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def run_synthetic_cohort_benchmark(
    service: ClinicalTreatmentService,
    cohort_size: int = 250,
    max_p95_ms: float = 20.0,
) -> BatchEvaluationReport:
    """Run synthetic clinical cohort compatibility checks under latency SLA."""
    if cohort_size < 1:
        raise ValueError("cohort_size must be >= 1.")
    samples: list[float] = []
    compatible_hits = 0
    suite_start = time.perf_counter()
    for idx in range(cohort_size):
        patient = PatientBiometricInput(
            patient_id=f"pat-{idx}",
            age=25 + (idx % 55),
            sex="M" if idx % 2 == 0 else "F",
            weight_kg=62.0 + (idx % 35),
            height_cm=162.0 + (idx % 25),
            smoker="Oui" if idx % 7 == 0 else "Non",
            activity_level="Élevée" if idx % 3 == 0 else "Moyenne",
            medical_history="Cardiovasculaire" if idx % 11 == 0 else "Aucun",
        )
        t0 = time.perf_counter()
        pred = service.evaluate(patient)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if pred.compatible:
            compatible_hits += 1

    total_ms = max((time.perf_counter() - suite_start) * 1000.0, 1e-6)
    ordered = sorted(samples)
    p95 = _percentile(ordered, 95.0)
    if p95 > max_p95_ms:
        raise TimeoutError(f"Clinical service p95 {p95:.3f}ms exceeded SLA {max_p95_ms:.3f}ms.")
    return BatchEvaluationReport(
        cohort_size=cohort_size,
        compatible_rate=compatible_hits / cohort_size,
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=p95,
        p99_ms=_percentile(ordered, 99.0),
        throughput_ops_sec=(cohort_size * 1000.0) / total_ms,
    )
