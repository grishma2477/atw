"""Screening inference and persistence for NeuroSense's doctor Diagnostics page."""

import json
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from math import isfinite
from pathlib import Path

from flask import current_app, jsonify, request, session

FEATURES = ("SES", "Age", "eTIV", "MMSE", "nWBV", "ASF", "Sex")
SYMPTOMS = {
    "tremors", "balance problems", "coordination issues", "stiffness",
    "slowed movement", "vision changes", "headaches", "dizziness",
}


@lru_cache(maxsize=2)
def load_model(directory):
    import joblib
    import xgboost as xgb

    scaler = joblib.load(directory / "dementia_scaler.pkl")
    if tuple(getattr(scaler, "feature_names_in_", ())) != FEATURES:
        raise ValueError("Unexpected OASIS scaler feature order")
    model = xgb.Booster()
    model.load_model(str(directory / "dementia_oasis_model.json"))
    if model.num_features() != len(FEATURES):
        raise ValueError("Unexpected OASIS model feature count")
    return scaler, model


def parse_features(data):
    values = {}
    for key in FEATURES:
        value = data.get(key)
        if value is None or isinstance(value, bool):
            raise ValueError(f"Enter a number for {key}.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Enter a number for {key}.") from None
        if not isfinite(number):
            raise ValueError(f"Enter a finite number for {key}.")
        values[key] = number

    if not (0 <= values["MMSE"] <= 30 and values["Sex"] in (0, 1)):
        raise ValueError("Enter MMSE from 0 to 30 and select a training sex code.")
    return values


def score_oasis(directory, values):
    import pandas as pd
    import xgboost as xgb

    scaler, model = load_model(directory)
    frame = pd.DataFrame([values], columns=list(FEATURES))
    prediction = model.predict(xgb.DMatrix(scaler.transform(frame)))
    if prediction.shape != (1,):
        raise ValueError("Expected one OASIS score")
    score = float(prediction[0])
    if not isfinite(score) or not 0 <= score <= 1:
        raise ValueError("OASIS score is outside the expected range")
    return score * 100


def questionnaire_points(raw):
    try:
        values = raw if isinstance(raw, dict) else json.loads(raw)
        if not isinstance(values, dict):
            return None
        age = float(values["age"])
        memory = str(values["memoryComplaints"]).strip().lower()
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    if not (isfinite(age) and 18 <= age <= 100):
        return None
    if memory not in {"none", "mild", "moderate", "severe"}:
        return None

    symptoms = values.get("neurologicalSymptoms")
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if not isinstance(symptoms, list) or not symptoms:
        return None
    if any(not isinstance(item, str) for item in symptoms):
        return None
    selected = {item.strip().lower() for item in symptoms}
    if selected == {"none"}:
        selected.clear()
    elif not selected <= SYMPTOMS:
        return None

    return (30 if age >= 70 else 0) + (
        25 if memory in {"moderate", "severe"} else 0
    ) + 5 * len(selected)


def save_screening(connection, doctor_id, patient_id, output):
    with closing(connection.cursor()) as cursor:
        try:
            connection.begin()
            cursor.execute(
                """
                INSERT INTO screening_reports (patient_id, doctor_id, result_json)
                VALUES (%s, %s, %s)
                """,
                (patient_id, doctor_id, json.dumps(output)),
            )
            report_id = cursor.lastrowid
            cursor.execute(
                """
                INSERT INTO health_events
                (patient_id, event_type, title, description, severity, disease, created_at)
                VALUES (%s, 'screening', %s, %s, 'low', NULL, NOW())
                """,
                (
                    patient_id,
                    "Screening report recorded",
                    f"Screening report #{report_id}. Source: {output['source']}.",
                ),
            )
            cursor.execute(
                """
                INSERT INTO audit_log
                (user_id, action, table_name, record_id, changes, created_at)
                VALUES (%s, 'CREATE', 'screening_reports', %s, %s, NOW())
                """,
                (doctor_id, report_id, json.dumps({"source": output["source"]})),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return report_id


def run_screening(mysql):
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return jsonify(status="error", message="Sign in as a doctor."), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(status="error", message="Send a JSON object."), 400
    patient_id = data.get("patient_id")
    if type(patient_id) is not int or patient_id <= 0:
        return jsonify(status="error", message="Select a patient."), 400

    try:
        features = parse_features(data)
    except ValueError as error:
        return jsonify(status="error", message=str(error)), 400

    connection = mysql.connection
    try:
        with closing(connection.cursor()) as cursor:
            cursor.execute("SELECT id FROM patients WHERE id = %s", (patient_id,))
            if cursor.fetchone() is None:
                return jsonify(status="error", message="Patient not found."), 404

        directory = Path(current_app.root_path) / "ml" / "Models"
        try:
            score = score_oasis(directory, features)
        except Exception:
            current_app.logger.exception("OASIS inference unavailable")
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    """
                    SELECT form_data FROM assessments WHERE patient_id = %s
                    ORDER BY created_at DESC, id DESC LIMIT 1
                    """,
                    (patient_id,),
                )
                latest = cursor.fetchone()
            points = questionnaire_points(latest["form_data"]) if latest else None
            if points is None:
                return jsonify(
                    status="error",
                    message="Model unavailable and no complete questionnaire is available.",
                ), 503
            output = {"source": "Questionnaire fallback", "questionnaire_points": points}
        else:
            band = "High" if score >= 70 else "Moderate" if score >= 40 else "Low"
            output = {
                "source": "OASIS XGBoost",
                "score": score,
                "prototype_band": band,
                "model_inputs": features,
            }

        report_id = save_screening(connection, doctor_id, patient_id, output)
        return jsonify(
            status="success",
            report_id=report_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **{key: value for key, value in output.items() if key != "model_inputs"},
        )
    except Exception:
        current_app.logger.exception("Could not complete screening request")
        return jsonify(status="error", message="The report could not be saved."), 500
