"""Rule-based questionnaire scoring for the current NeuroSense forms."""

from dataclasses import dataclass


@dataclass
class RiskPrediction:
    disease: str
    score: float
    confidence: float
    risk_level: str
    top_factors: list[list]


@dataclass
class DiagnosticResult:
    primary_diagnosis: str
    diagnosis_confidence: float
    secondary_diagnoses: list[str]
    disease_probabilities: dict[str, float]
    key_findings: list[str]
    recommendations: list[str]


class MLModelManager:
    # Retained for callers that check whether trained inference is active.
    fallback_mode = True

    @staticmethod
    def _questionnaire_factors(form):
        factors = []

        try:
            age = int(form.get("age"))
        except (TypeError, ValueError):
            age = None

        if age is not None and 70 <= age <= 120:
            factors.append(["Age", 30])

        memory = str(form.get("memoryComplaints") or "").strip().casefold()
        if memory in {"moderate", "severe"}:
            factors.append(["Memory complaints", 25])

        submitted_symptoms = form.get("neurologicalSymptoms")
        if isinstance(submitted_symptoms, str):
            submitted_symptoms = [submitted_symptoms]
        elif not isinstance(submitted_symptoms, (list, tuple)):
            submitted_symptoms = []

        symptoms = {
            item.strip().casefold()
            for item in submitted_symptoms
            if isinstance(item, str)
            and item.strip()
            and item.strip().casefold() != "none"
        }

        if symptoms:
            factors.append(["Reported neurological symptoms", 5 * len(symptoms)])

        return factors

    def _fallback_risk(self, disease, form):
        factors = self._questionnaire_factors(form)

        return RiskPrediction(
            disease=disease,
            score=float(sum(points for _, points in factors)),
            confidence=0.0,
            risk_level="unavailable",
            top_factors=factors,
        )

    def predict_alzheimers_risk(self, form):
        return self._fallback_risk("Alzheimer's", form)

    def predict_parkinsons_risk(self, form):
        return self._fallback_risk("Parkinson's", form)

    def predict_dementia_risk(self, form):
        return self._fallback_risk("Dementia", form)

    def predict_diagnostic(self, form):
        return DiagnosticResult(
            primary_diagnosis="No diagnostic result available",
            diagnosis_confidence=0.0,
            secondary_diagnoses=[],
            disease_probabilities={},
            key_findings=[
                "The current form does not supply the inputs required by "
                "the trained diagnostic models."
            ],
            recommendations=[],
        )

    def _fallback_diagnostic(self, form):
        # Compatibility for older routes that still call this method.
        return self.predict_diagnostic(form)


_model_manager = MLModelManager()


def get_model_manager():
    return _model_manager