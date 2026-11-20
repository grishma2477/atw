"""Patient assessment workflow and persistence."""

import json
from contextlib import closing

from flask import abort, redirect, render_template, request, session, url_for


STEP_FIELDS = {
    1: ("age", "gender", "weight", "height", "education", "occupation", "languagePreference"),
    2: ("familyHistory", "medicalConditions", "medications", "allergies"),
    3: (
        "physicalActivity", "sleepQuality", "alcoholConsumption", "smokingStatus",
        "dietType", "coffeeConsumption", "socialEngagement",
    ),
    4: ("memoryComplaints", "speechIssues", "neurologicalSymptoms", "moodChanges"),
}

MULTI_SELECT_FIELDS = {
    "familyHistory", "medicalConditions", "medications", "neurologicalSymptoms",
}


def save_assessment(connection, patient_id, answers, risk_scores, diagnostic):
    """Save the submitted answers and model output together."""
    with closing(connection.cursor()) as cursor:
        try:
            cursor.execute(
                """
                INSERT INTO assessments (patient_id, form_data, alz, park, dem, diag)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    patient_id, json.dumps(answers),
                    *(json.dumps(score) for score in risk_scores),
                    json.dumps(diagnostic),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def register_assessment_routes(app, mysql, get_model_manager):
    @app.route("/assessment", methods=["GET", "POST"])
    def assessment():
        patient_id = session.get("patient_id")
        if patient_id is None:
            return redirect(url_for("patient_login"))

        step = request.args.get("step", "1")
        if step not in ("1", "2", "3", "4"):
            abort(400, description="Invalid assessment step.")
        step = int(step)

        draft = session.get("assessment_draft") or {}
        if draft.get("patient_id") != patient_id:
            draft = {"patient_id": patient_id, "answers": {}}
        answers = draft["answers"].copy()

        if request.method == "GET":
            return render_template(f"assessment_step{step}.html", step=step, answers=answers)

        for field in STEP_FIELDS[step]:
            answers.pop(field, None)
            if field in MULTI_SELECT_FIELDS:
                answers[field] = request.form.getlist(field)
            elif field in request.form:
                answers[field] = request.form[field]

        # Assign the top-level key so the session saves each submitted step.
        session["assessment_draft"] = {"patient_id": patient_id, "answers": answers}

        if step < 4:
            return redirect(url_for("assessment", step=step + 1))

        # Preserve the model's previous scalar input contract. The database
        # retains every checkbox selection for use by compatible consumers.
        model_input = answers.copy()
        for field in MULTI_SELECT_FIELDS:
            if field in model_input:
                selections = model_input.pop(field)
                if selections:
                    model_input[field] = selections[0]

        model = get_model_manager()
        risk_scores = [
            vars(model.predict_alzheimers_risk(model_input)),
            vars(model.predict_parkinsons_risk(model_input)),
            vars(model.predict_dementia_risk(model_input)),
        ]
        diagnostic = vars(model.predict_diagnostic(model_input))

        save_assessment(mysql.connection, patient_id, answers, risk_scores, diagnostic)
        session.pop("assessment_draft", None)

        return render_template(
            "assessment_result.html", risk_scores=risk_scores, diagnostic=diagnostic,
        )

    @app.get("/patient_assessment")
    def patient_assessment():
        return redirect(url_for("assessment", step=1))
