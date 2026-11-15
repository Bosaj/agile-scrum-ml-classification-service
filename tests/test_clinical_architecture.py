"""Comprehensive clinical validation, DI, telemetry, and synthetic cohort load tests."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clinical_ml.contracts import (
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
    JsonLogFormatter,
    check_liveness,
    check_readiness,
    get_health_status,
    run_synthetic_cohort_benchmark,
)


class _StubClassifier:
    def predict_probability(self, features: FeatureVector) -> float:
        return 0.88


class TestClinicalArchitecture(unittest.TestCase):
    def test_biometric_validation_and_bmi_calculation(self) -> None:
        patient = PatientBiometricInput(
            patient_id="p-001",
            age=35,
            sex="M",
            weight_kg=74.5,
            height_cm=172.0,
        )
        self.assertAlmostEqual(patient.bmi, 25.18, places=2)

        with self.assertRaises(ValueError):
            PatientBiometricInput(patient_id=" ", age=35, sex="M", weight_kg=70.0, height_cm=170.0)
        with self.assertRaises(ValueError):
            PatientBiometricInput(patient_id="p", age=150, sex="M", weight_kg=70.0, height_cm=170.0)
        with self.assertRaises(ValueError):
            PatientBiometricInput(patient_id="p", age=30, sex="X", weight_kg=70.0, height_cm=170.0)
        with self.assertRaises(ValueError):
            PatientBiometricInput(patient_id="p", age=30, sex="F", weight_kg=math.nan, height_cm=170.0)
        with self.assertRaises(ValueError):
            PatientBiometricInput(
                patient_id="p",
                age=30,
                sex="F",
                weight_kg=60.0,
                height_cm=165.0,
                allergy_type="Invalid",
            )

    def test_risk_stratification_low_vs_high_risk_profiles(self) -> None:
        telemetry = InMemoryTelemetrySink()
        service = ClinicalTreatmentService.create_default(telemetry=telemetry)

        healthy = service.evaluate(
            PatientBiometricInput(
                patient_id="p-low",
                age=28,
                sex="F",
                weight_kg=60.0,
                height_cm=168.0,
                activity_level="Élevée",
            )
        )
        high_risk = service.evaluate(
            PatientBiometricInput(
                patient_id="p-high",
                age=72,
                sex="M",
                weight_kg=120.0,
                height_cm=168.0,
                smoker="Oui",
                alcohol="Oui",
                activity_level="Faible",
                medical_history="Cardiovasculaire",
                allergy_type="Médicamenteuse",
            )
        )
        self.assertTrue(healthy.compatible)
        self.assertGreater(healthy.compatibility_probability, high_risk.compatibility_probability)
        self.assertEqual(high_risk.risk_band, "HIGH_RISK")
        self.assertEqual(len(telemetry.events), 2)

    def test_dependency_injection_and_observability_probes(self) -> None:
        container = ClinicalContainer()
        recorder = InMemoryTelemetrySink()
        container.register_instance(TelemetrySink, recorder)
        container.register_instance(TreatmentClassifierProtocol, _StubClassifier())

        service = ClinicalTreatmentService(
            encoder=ClinicalFeatureEncoder(),
            classifier=container.resolve(TreatmentClassifierProtocol),
            telemetry=container.resolve(TelemetrySink),
        )
        res = service.evaluate(
            PatientBiometricInput(patient_id="p-di", age=40, sex="M", weight_kg=75.0, height_cm=175.0)
        )
        self.assertAlmostEqual(res.compatibility_probability, 0.88)
        self.assertEqual(res.risk_band, "LOW_RISK")

        self.assertTrue(check_liveness())
        self.assertTrue(check_readiness())
        self.assertEqual(get_health_status()["status"], "UP")

        log_rec = logging.LogRecord("clinical", logging.INFO, __file__, 1, "clinical probe", (), None)
        self.assertEqual(json.loads(JsonLogFormatter().format(log_rec))["message"], "clinical probe")

    def test_synthetic_cohort_benchmark_sla(self) -> None:
        service = ClinicalTreatmentService.create_default()
        report = run_synthetic_cohort_benchmark(service=service, cohort_size=250, max_p95_ms=20.0)
        self.assertEqual(report.cohort_size, 250)
        self.assertLess(report.p95_ms, 20.0)
        self.assertGreater(report.throughput_ops_sec, 500.0)


if __name__ == "__main__":
    unittest.main()
