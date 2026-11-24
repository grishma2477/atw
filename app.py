import os
from pathlib import Path
from functools import lru_cache
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
import json
import re
from contextlib import closing
from math import isfinite

from flask import abort

# Set TensorFlow environment variables BEFORE importing ML modules
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import (
    Flask, render_template, request, redirect,
    session, url_for, jsonify, send_file, flash
)

from cachelib.file import FileSystemCache
from flask_session import Session
from flask_mysqldb import MySQL

from assessment_routes import register_assessment_routes
from ml.ml_manager import get_model_manager

from datetime import datetime
import traceback
import json

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import io

from neurosense_diagnostics import run_screening



app = Flask(__name__)
app.secret_key = "SECRET123"

# ----------------------
# MySQL CONFIG
# ----------------------
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "root"
app.config["MYSQL_DB"] = "neurosense"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

# Keep assessment drafts on the server. The cookie carries a session ID.
session_dir = Path(app.instance_path) / "sessions"
session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
app.config.update(
    SESSION_TYPE="cachelib",
    SESSION_CACHELIB=FileSystemCache(str(session_dir), mode=0o600),
    SESSION_PERMANENT=False,
    SESSION_COOKIE_SECURE=os.getenv("NEUROSENSE_HTTPS") == "1",
    SESSION_COOKIE_SAMESITE="Lax",
)
Session(app)
mysql = MySQL(app)

# ---------------------------
# HOME PAGE
# ---------------------------
@app.route("/")
def index():
    return render_template("patient_login.html")

@app.route("/check")
def check():
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT id, full_name
        FROM patients
        WHERE full_name IS NOT NULL AND full_name != ''
        ORDER BY full_name
    """)
    patients = cur.fetchall()

    return render_template(
        "check.html",
        patients=patients
    )
   

# ---------------------------
# PATIENT SIGNUP
# ---------------------------
@app.route("/patient_signup", methods=["GET","POST"])
def patient_signup():
    if request.method=="POST":
        name=request.form["name"]
        email=request.form["email"]
        password=request.form["password"]

        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO patients(full_name,email,password) VALUES(%s,%s,%s)", (name,email,password))
        mysql.connection.commit()

        return redirect(url_for("patient_login"))
    return render_template("patient_signup.html")
    
@app.route("/doctor_register", methods=["GET","POST"])
def doctor_register():
    if request.method=="POST":
        name=request.form["name"]
        email=request.form["email"]
        password=request.form["password"]

        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO `doctors`( `name`, `email`, `password`)   VALUES(%s,%s,%s)", (name,email,password))
        mysql.connection.commit()

        return redirect(url_for("doctor_login"))
    return render_template("doctor_signup.html")

# ---------------------------
# PATIENT LOGIN
# ---------------------------
@app.route("/patient_login", methods=["GET","POST"])
def patient_login():
    if request.method=="POST":
        email=request.form["email"]
        password=request.form["password"]

        cur=mysql.connection.cursor()
        cur.execute("SELECT * FROM patients WHERE email=%s AND password=%s", (email, password))
        user=cur.fetchone()

        if user:
            session["patient_id"]=user["id"]
            session["patient"] = "Patient User"
            return redirect(url_for("patient_dashboard"))
        return "Invalid login"
    return render_template("patient_login.html")

# ---------------------------
# PATIENT DASHBOARD
# ---------------------------

PROFILE_GENDERS = ("Female", "Male", "Other")
PROFILE_BLOOD_TYPES = ("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-")


def load_patient_profile(patient_id):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            """
            SELECT full_name, email, phone, age, gender, blood_type
            FROM patients
            WHERE id = %s
            """,
            (patient_id,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()

# ==========================================
# UPDATE PROFILE (SAVE CHANGES)
# ==========================================

SCREENING_REFERENCE = re.compile(r"^Screening report #([1-9][0-9]*)\.")


def read_json_object(value):
    if isinstance(value, dict):
        return value

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def explain_screening(raw):
    result = read_json_object(raw)
    score = result.get("score")
    points = result.get("questionnaire_points")

    if (
        result.get("source") == "OASIS XGBoost"
        and type(score) in (int, float)
        and 0 <= score <= 100
        and isfinite(score)
    ):
        if score >= 70:
            meaning = (
                "The information entered was similar to patterns linked "
                "to dementia in the model’s research data."
            )
        elif score >= 40:
            meaning = (
                "The information entered gave a score in the middle "
                "range of this research model."
            )
        else:
            meaning = (
                "The information entered gave a score in the lower "
                "range of this research model."
            )

        return {
            "badge": f"Research score: {score:.2f}%",
            "explanation": (
                f"{meaning} This score describes the computer’s result "
                "for the information entered. Your doctor can review it "
                "alongside your health history and other assessments."
            ),
        }

    if (
        result.get("source") == "Questionnaire fallback"
        and type(points) is int
        and points >= 0
    ):
        return {
            "badge": f"Questionnaire score: {points} points",
            "explanation": (
                "The app added points from the saved questionnaire answers "
                "about age, memory and symptoms. This is a summary of the "
                "answers provided."
            ),
        }

    return {
        "badge": None,
        "explanation": "A score is unavailable for this screening entry.",
    }


def attach_event_explanations(cursor, patient_id, events):
    report_ids = set()

    for event in events:
        report_id = None

        if event["event_type"] == "screening":
            match = SCREENING_REFERENCE.match(event["description"] or "")
            if match:
                report_id = int(match.group(1))
                report_ids.add(report_id)

        event["screening_report_id"] = report_id

    reports = {}
    if report_ids:
        placeholders = ", ".join(["%s"] * len(report_ids))

        cursor.execute(
            f"""
            SELECT id, result_json
            FROM screening_reports
            WHERE patient_id = %s AND id IN ({placeholders})
            """,
            (patient_id, *sorted(report_ids)),
        )
        reports = {row["id"]: row["result_json"] for row in cursor.fetchall()}

    for event in events:
        event_type = event["event_type"]

        if event_type == "screening":
            event.update(
                explain_screening(reports.get(event["screening_report_id"]))
            )
        else:
            event["badge"] = None
            event["explanation"] = {
                "diagnosis": "This entry contains a diagnosis-related note. Your doctor can explain what was recorded.",
                "intervention": "This entry contains a follow-up plan or recommendation recorded by your care team.",
            }.get(
                event_type,
                "This entry records an update in your health timeline.",
            )

@app.route("/timeline")
def patient_timeline():
    if "patient_id" not in session:
        return redirect(url_for("patient_login"))

    pid = session["patient_id"]
    event_type = request.args.get("type")
    disease = request.args.get("disease")

    query = """
        SELECT title, description, disease, event_type, created_at
        FROM health_events
        WHERE patient_id = %s
    """
    params = [pid]

    if event_type:
        query += " AND event_type = %s"
        params.append(event_type)

    if disease:
        query += " AND disease = %s"
        params.append(disease)

    query += " ORDER BY created_at DESC"

    with closing(mysql.connection.cursor()) as cur:
        cur.execute(query, params)
        events = list(cur.fetchall())
        attach_event_explanations(cur, pid, events)

    return render_template(
        "patient_timeline.html",
        events=events,
        active_page="timeline",
    )


@app.route("/timeline_export")
def export_timeline():
    if "patient_id" not in session:
        return redirect(url_for("patient_login"))

    pid = session["patient_id"]
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT title, description, disease, created_at
        FROM health_events
        WHERE patient_id=%s
        ORDER BY created_at DESC
    """, (pid,))
    events = cur.fetchall()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Health Timeline")
    y -= 40

    pdf.setFont("Helvetica", 11)
    for e in events:
        pdf.drawString(50, y, f"{e['created_at']} - {e['title']}")
        y -= 15
        pdf.drawString(60, y, e['description'])
        y -= 20
        pdf.drawString(60, y, f"Disease: {e['disease']}")
        y -= 30

        if y < 100:
            pdf.showPage()
            y = 800

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Health_Timeline.pdf",
        mimetype="application/pdf"
    )



# ==========================================
# MY EHR PAGE
# ==========================================
def ehr_reported_items(value):
    """Read both older single selections and newer checkbox lists."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def ehr_json_object(value):
    """Read an assessment JSON column as a dictionary."""
    if isinstance(value, dict):
        return value

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


@app.route("/ehr")
def patient_ehr():
    patient_id = session.get("patient_id")
    if patient_id is None:
        return redirect(url_for("patient_login"))

    tabs = {
        "medications": "Medications",
        "allergies": "Allergies",
        "history": "Medical History",
        "assessments": "Assessments",
    }
    selected_tab = request.args.get("tab", "assessments")
    if selected_tab not in tabs:
        selected_tab = "assessments"

    cursor = mysql.connection.cursor()

    try:
        cursor.execute(
            """
            SELECT full_name, age, gender, blood_type, email, phone
            FROM patients
            WHERE id = %s
            """,
            (patient_id,),
        )
        info = cursor.fetchone()

        if info is None:
            session.clear()
            return redirect(url_for("patient_login"))

        cursor.execute(
            """
            SELECT diag, created_at, form_data
            FROM assessments
            WHERE patient_id = %s
            ORDER BY created_at DESC
            """,
            (patient_id,),
        )
        assessments = cursor.fetchall()
    finally:
        cursor.close()

    latest_form = ehr_json_object(assessments[0]["form_data"]) if assessments else {}

    for assessment in assessments:
        assessment["diag"] = ehr_json_object(assessment["diag"])
        assessment["form_data"] = ehr_json_object(assessment["form_data"])

    return render_template(
        "patient_ehr.html",
        active_page="ehr",
        info=info,
        tabs=tabs,
        selected_tab=selected_tab,
        assessments=assessments,
        medications=ehr_reported_items(latest_form.get("medications")),
        allergies=ehr_reported_items(latest_form.get("allergies")),
        conditions=ehr_reported_items(latest_form.get("medicalConditions")),
        family_history=ehr_reported_items(latest_form.get("familyHistory")),
    )

# ==========================================
# DOWNLOAD FULL EHR
# ==========================================
@app.route("/ehr_download")
def download_ehr():
    if "patient_id" not in session:
        return redirect(url_for("patient_login"))

    pid = session["patient_id"]
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM patients WHERE id=%s", (pid,))
    patient = cur.fetchone()

    cur.execute("SELECT diag, created_at FROM assessments WHERE patient_id=%s", (pid,))
    assessments = cur.fetchall()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Electronic Health Record")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Name: {patient['full_name']}")
    y -= 15
    pdf.drawString(50, y, f"Email: {patient['email']}")
    y -= 15
    pdf.drawString(50, y, f"Age: {patient['age']}")
    y -= 30

    for a in assessments:
        d = json.loads(a["diag"])
        pdf.drawString(50, y, f"Diagnosis: {d['primary_diagnosis']}")
        y -= 15
        pdf.drawString(60, y, f"Confidence: {d['diagnosis_confidence']}")
        y -= 25

        if y < 100:
            pdf.showPage()
            y = 800

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="EHR_Report.pdf",
        mimetype="application/pdf"
    )


# ==========================================
# PATIENT PROFILE PAGE
# ==========================================
@app.route("/patient_profile")
def patient_profile():
    patient_id = session.get("patient_id")
    if patient_id is None:
        return redirect(url_for("patient_login"))

    info = load_patient_profile(patient_id)
    if info is None:
        session.clear()
        return redirect(url_for("patient_login"))

    return render_template(
        "patient_profile.html",
        active_page="profile",
        info=info,
        genders=PROFILE_GENDERS,
        blood_types=PROFILE_BLOOD_TYPES,
    )

@app.route("/update", methods=["POST"])
def update_profile():
    patient_id = session.get("patient_id")
    if patient_id is None:
        return redirect(url_for("patient_login"))

    full_name = request.form.get("full_name", "").strip()
    phone = request.form.get("phone", "").strip()
    age = request.form.get("age", type=int)
    gender = request.form.get("gender")
    blood_type = request.form.get("blood_type")

    if (
        not full_name
        or age is None
        or not 1 <= age <= 120
        or gender not in PROFILE_GENDERS
        or blood_type not in PROFILE_BLOOD_TYPES
    ):
        flash("Check your name, age, gender, and blood type.", "profile-error")
        return redirect(url_for("patient_profile"))

    connection = mysql.connection
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            UPDATE patients
            SET full_name = %s,
                phone = %s,
                age = %s,
                gender = %s,
                blood_type = %s
            WHERE id = %s
            """,
            (full_name, phone, age, gender, blood_type, patient_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        flash("Your changes couldn’t be saved. Please try again.", "profile-error")
    else:
        flash("Your profile has been saved.", "profile-success")
    finally:
        cursor.close()

    return redirect(url_for("patient_profile"))


@app.route("/patient_dashboard")
def patient_dashboard():
    patient_id = session.get("patient_id")
    if patient_id is None:
        return redirect(url_for("patient_login"))

    info = load_patient_profile(patient_id)
    if info is None:
        session.clear()
        return redirect(url_for("patient_login"))

    name = (info["full_name"] or "").strip()
    age = info["age"]
    profile_incomplete = not (
        name
        and age is not None
        and 1 <= age <= 120
        and info["gender"] in PROFILE_GENDERS
        and info["blood_type"] in PROFILE_BLOOD_TYPES
    )

    return render_template(
        "patient_dashboard.html",
        active_page="dashboard",
        patient=name,
        profile_incomplete=profile_incomplete,
    )

register_assessment_routes(app, mysql, get_model_manager)

# ---------------------------
# DOCTOR LOGIN
# ---------------------------
@app.route("/doctor_login", methods=["GET","POST"])
def doctor_login():
    if request.method=="POST":
        email=request.form["email"]
        password=request.form["password"]

        cur=mysql.connection.cursor()
        cur.execute("SELECT * FROM doctors WHERE email=%s AND password=%s", (email,password))
        doc=cur.fetchone()

        if doc:
            session["doctor_id"]=doc["id"]
            return redirect(url_for("doctor_dashboard"))
        return "Invalid doctor login"
    return render_template("doctor_login.html")

# ---------------------------
# DOCTOR DASHBOARD
# ---------------------------
@app.route("/doctor_dashboard")
def doctor_dashboard():
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    cur = mysql.connection.cursor()

    # Total patients
    cur.execute("SELECT COUNT(*) AS total FROM patients")
    total_patients = cur.fetchone()["total"]

    # High risk patients
    cur.execute("""
        SELECT COUNT(*) AS high_risk
        FROM diagnostic_reports
        WHERE diagnosis_confidence >= 0.7
    """)
    high_risk = cur.fetchone()["high_risk"]

    # Pending cases
    cur.execute("""
        SELECT COUNT(*) AS pending
        FROM diagnostic_reports
        WHERE diagnosis_confidence < 0.7
    """)
    pending = cur.fetchone()["pending"]

    # Alerts (high severity events)
    cur.execute("""
        SELECT COUNT(*) AS alerts
        FROM health_events
        WHERE severity = 'high'
    """)
    alerts = cur.fetchone()["alerts"]

    # Patient roster
    cur.execute(
        """
        SELECT p.id, p.full_name, p.age, p.gender
        FROM patients p
        JOIN doctor_patient_access a ON a.patient_id = p.id
        WHERE a.doctor_id = %s
        ORDER BY p.id DESC
        LIMIT 5
        """,
        (session["doctor_id"],),
    )
    patients = cur.fetchall()

    # Recent activity
    cur.execute("""
        SELECT title, created_at
        FROM health_events
        ORDER BY created_at DESC
        LIMIT 5
    """)
    activities = cur.fetchall()

    return render_template(
        "doctor_dashboard.html",
        total_patients=total_patients,
        high_risk=high_risk,
        pending=pending,
        alerts=alerts,
        patients=patients,
        activities=activities
    )

# ---------------------------
# DOCTOR - PATIENT MANAGEMENT
# ---------------------------
@app.route("/doctor_patients")
def doctor_patients():
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return redirect(url_for("doctor_login"))

    search = request.args.get("search", "").strip()

    query = """
        SELECT p.id, p.full_name, p.email, p.age, p.gender
        FROM patients AS p
        JOIN doctor_patient_access AS access
            ON access.patient_id = p.id
        WHERE access.doctor_id = %s
    """
    params = [doctor_id]

    if search:
        query += " AND (p.full_name LIKE %s OR p.email LIKE %s)"
        search_term = f"%{search}%"
        params.extend([search_term, search_term])

    query += " ORDER BY p.id DESC"

    cur = mysql.connection.cursor()
    try:
        cur.execute(query, tuple(params))
        patients = cur.fetchall()
    finally:
        cur.close()

    return render_template(
        "patient_management.html",
        patients=patients,
        search=search,
    )


@app.route("/doctor_diagnostics_dash")
def doctor_diagnostics_page():
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT id, full_name
        FROM patients
        WHERE full_name IS NOT NULL AND full_name != ''
        ORDER BY full_name
    """)
    patients = cur.fetchall()

    return render_template(
        "diagnostics.html",
        patients=patients
    )



def insert_diagnostic_report(data):
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO diagnostic_reports
        (patient_id, doctor_id, primary_diagnosis, diagnosis_confidence,
         secondary_diagnoses, disease_probabilities, key_findings, recommendations,
         mmse_score, cdr_score, csf_tau_level, csf_abeta42_level, apoe_status,
         mri_file_path, created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """, (
        data["patient_id"],
        data["doctor_id"],
        data["primary_diagnosis"],
        data["diagnosis_confidence"],
        json.dumps(data["secondary_diagnoses"]),
        json.dumps(data["disease_probabilities"]),
        json.dumps(data["key_findings"]),
        json.dumps(data["recommendations"]),
        data["mmse_score"],
        data["cdr_score"],
        data["csf_tau_level"],
        data["csf_abeta42_level"],
        data["apoe_status"],
        data["mri_file_path"]
    ))
    mysql.connection.commit()
    return cur.lastrowid

# -------------------------------------------------------------------
# MAIN ROUTE — RUN DIAGNOSTICS

def doctor_can_view_patient(cursor, doctor_id, patient_id):
    cursor.execute(
        """
        SELECT 1
        FROM doctor_patient_access
        WHERE doctor_id = %s AND patient_id = %s
        """,
        (doctor_id, patient_id),
    )
    return cursor.fetchone() is not None


@app.route("/doctor/patients/<int:patient_id>")
def doctor_patient_detail(patient_id):
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return redirect(url_for("doctor_login"))

    with closing(mysql.connection.cursor()) as cur:
        if not doctor_can_view_patient(cur, doctor_id, patient_id):
            abort(404)

        cur.execute(
            """
            SELECT id, full_name, email, phone, age, gender, blood_type
            FROM patients
            WHERE id = %s
            """,
            (patient_id,),
        )
        patient = cur.fetchone()
        if patient is None:
            abort(404)

        cur.execute(
            """
            SELECT created_at, form_data
            FROM assessments
            WHERE patient_id = %s
            ORDER BY created_at DESC
            """,
            (patient_id,),
        )
        assessments = [
            {
                "created_at": row["created_at"],
                "answers": read_json_object(row["form_data"]),
            }
            for row in cur.fetchall()
        ]

        cur.execute(
            """
            SELECT id, created_at, result_json
            FROM screening_reports
            WHERE patient_id = %s
            ORDER BY created_at DESC, id DESC
            """,
            (patient_id,),
        )
        screenings = [
            {**row, **explain_screening(row["result_json"])}
            for row in cur.fetchall()
        ]

        cur.execute(
            """
            SELECT title, description, disease, event_type, created_at
            FROM health_events
            WHERE patient_id = %s
            ORDER BY created_at DESC
            """,
            (patient_id,),
        )
        events = cur.fetchall()

    return render_template(
        "doctor_patient_detail.html",
        patient=patient,
        assessments=assessments,
        screenings=screenings,
        events=events,
    )

# Add near the top of app2.py if it isn't already imported:
from contextlib import closing


@app.route("/doctor_patient_latest-assessment/<int:patient_id>")
def get_latest_assessment(patient_id):
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return jsonify(status="error", message="Unauthorized"), 401

    with closing(mysql.connection.cursor()) as cur:
        if not doctor_can_view_patient(cur, doctor_id, patient_id):
            return jsonify(status="error", message="Patient not found."), 404

        cur.execute("""
            SELECT diag
            FROM assessments
            WHERE patient_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """, (patient_id,))
        row = cur.fetchone()

    if row is None:
        return jsonify(status="empty")

    diag = read_json_object(row["diag"])

    def display_text(value):
        if isinstance(value, list):
            return ", ".join(item for item in value if isinstance(item, str))
        return value if isinstance(value, str) else ""

    return jsonify(
        status="success",
        primary_diagnosis=display_text(diag.get("primary_diagnosis")),
        diagnosis_confidence=diag.get("diagnosis_confidence"),
        secondary_diagnoses=display_text(diag.get("secondary_diagnoses")),
        recommendations=display_text(diag.get("recommendations")),
        key_findings=display_text(diag.get("key_findings")),
    )
    if "doctor_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    cur = mysql.connection.cursor()
    if not doctor_can_view_patient(cur, session["doctor_id"], patient_id):
        cur.close()
        return jsonify({"status": "error", "message": "Patient not found."}), 404

    cur.execute("""
        SELECT diag
        FROM assessments
        WHERE patient_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (patient_id,))

    row = cur.fetchone()

    if not row:
        return jsonify({"status": "empty"})

    try:
        diag = json.loads(row["diag"])
    except:
        diag = {}

    return jsonify({
        "status": "success",
        "primary_diagnosis": diag.get("primary_diagnosis", ""),
        "diagnosis_confidence": diag.get("diagnosis_confidence", ""),
        "secondary_diagnoses": ", ".join(diag.get("secondary_diagnoses", [])),
        "recommendations": ", ".join(diag.get("recommendations", [])),
        "key_findings": ", ".join(diag.get("key_findings", []))
    })

@app.route("/doctor_diagnostics", methods=["POST"])
def run_diagnostics():
    return run_screening(mysql)

@app.route("/upload_mri", methods=["POST"])
def upload_mri():
    if "doctor_id" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if "mri_file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"})

    file = request.files["mri_file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"})

    upload_dir = "uploads/mri"
    os.makedirs(upload_dir, exist_ok=True)

    filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
    filepath = os.path.join(upload_dir, filename)

    file.save(filepath)

    return jsonify({
        "status": "success",
        "message": "MRI uploaded successfully",
        "file": filepath
    })


@app.route("/doctor_patients_add", methods=["GET", "POST"])
def add_patient():
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    if request.method == "POST":
        full_name = request.form["full_name"]
        email = request.form["email"]
        phone = request.form.get("phone")
        age = request.form.get("age")
        gender = request.form.get("gender")
        blood_type = request.form.get("blood_type")
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO patients
            (full_name, email, phone, age, gender, blood_type, password)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            full_name,
            email,
            phone,
            age,
            gender,
            blood_type,
            password
        ))
        new_patient_id = cur.lastrowid
        cur.execute(
            """
            INSERT INTO doctor_patient_access (doctor_id, patient_id)
            VALUES (%s, %s)
            """,
            (session["doctor_id"], new_patient_id),
        )
        mysql.connection.commit()

        return redirect(url_for("doctor_patients"))

    return render_template("add_patient.html")



# ---------------------------
# LOGOUT
# ---------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("patient_login"))

# ---------------------------
# DOCTOR - INTERVENTIONS
# ---------------------------
@app.route("/doctor_interventions", methods=["GET", "POST"])
def doctor_interventions():
    if "doctor_id" not in session:
        return redirect(url_for("doctor_login"))

    cur = mysql.connection.cursor()

    # -------------------------------
    # SAVE INTERVENTION
    # -------------------------------
    if request.method == "POST":
        patient_id = request.form["patient_id"]
        diagnosis = request.form["diagnosis"]
        intervention_type = request.form["intervention_type"]
        details = request.form["details"]

        cur.execute("""
            INSERT INTO health_events
            (patient_id, event_type, title, description, severity, disease, created_at)
            VALUES (%s,'intervention',%s,%s,'moderate',%s,NOW())
        """, (
            patient_id,
            f"{intervention_type} Intervention",
            details,
            diagnosis
        ))
        mysql.connection.commit()

        return redirect(url_for("doctor_interventions"))

    # -------------------------------
    # FETCH PATIENTS
    # -------------------------------
    cur.execute("""
        SELECT id, full_name
        FROM patients
        WHERE full_name IS NOT NULL AND full_name != ''
        ORDER BY full_name
    """)
    patients = cur.fetchall()

    # -------------------------------
    # FETCH INTERVENTIONS
    # -------------------------------
    cur.execute("""
        SELECT 
            he.id,
            p.full_name,
            he.title,
            he.description,
            he.disease,
            he.created_at
        FROM health_events he
        JOIN patients p ON he.patient_id = p.id
        WHERE he.event_type = 'intervention'
        ORDER BY he.created_at DESC
        LIMIT 10
    """)
    interventions = cur.fetchall()

    return render_template(
        "interventions.html",
        patients=patients,
        interventions=interventions
    )



@app.route("/add_note", methods=["POST"])
def run_diagnostics_flask():
    try:
        body = request.json
        model_manager = get_model_manager()

        patient_id = body.get("patient_id")
        doctor_id = body.get("doctor_id")

        print("=== Starting Diagnostics for Patient:", patient_id, "===")

        # -------------------------------
        # STEP 1 — MODEL PREDICTION
        # -------------------------------
        try:
            diagnostic_result = model_manager.predict_diagnostic(body)
        except Exception as e:
            print("⚠️ Model Error — Using fallback:", e)
            diagnostic_result = model_manager._fallback_diagnostic(body)

        if not diagnostic_result:
            diagnostic_result = model_manager._fallback_diagnostic(body)

        # -------------------------------
        # STEP 2 — INSERT DIAGNOSTIC REPORT INTO DB
        # -------------------------------
        report_id = None
        try:
            cur = mysql.connection.cursor()

            cur.execute("""
                INSERT INTO diagnostic_reports 
                (patient_id, doctor_id, primary_diagnosis, diagnosis_confidence,
                 secondary_diagnoses, disease_probabilities, key_findings, recommendations,
                 mmse_score, cdr_score, csf_tau_level, csf_abeta42_level, apoe_status, 
                 mri_file_path, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
            """, (
                patient_id,
                doctor_id,
                diagnostic_result.primary_diagnosis,
                diagnostic_result.diagnosis_confidence,
                json.dumps(diagnostic_result.secondary_diagnoses),
                json.dumps(diagnostic_result.disease_probabilities),
                json.dumps(diagnostic_result.key_findings),
                json.dumps(diagnostic_result.recommendations),
                body.get("mmse_score"),
                body.get("cdr_score"),
                body.get("csf_tau_level"),
                body.get("csf_abeta42_level"),
                body.get("apoe_status"),
                body.get("mri_file_path")
            ))

            mysql.connection.commit()
            report_id = cur.lastrowid

        except Exception as e:
            print("⚠️ DB Error inserting diagnostic report:", e)
            report_id = f"temp_{int(datetime.utcnow().timestamp())}"

        # -------------------------------
        # STEP 3 — INSERT HEALTH EVENT
        # -------------------------------
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO health_events
                (patient_id, event_type, title, description, severity, disease, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (
                patient_id,
                "diagnosis",
                f"Diagnostic Report: {diagnostic_result.primary_diagnosis}",
                f"Primary diagnosis: {diagnostic_result.primary_diagnosis}",
                "high" if diagnostic_result.diagnosis_confidence > 0.7 else "moderate",
                diagnostic_result.primary_diagnosis
            ))
            mysql.connection.commit()
        except Exception as e:
            print("⚠️ Error inserting health event:", e)

        # -------------------------------
        # STEP 4 — AUDIT LOG
        # -------------------------------
        try:
            cur = mysql.connection.cursor()
            cur.execute("""
                INSERT INTO audit_log
                (user_id, action, table_name, record_id, changes, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                doctor_id,
                "CREATE",
                "diagnostic_reports",
                report_id,
                json.dumps({"diagnosis": diagnostic_result.primary_diagnosis})
            ))
            mysql.connection.commit()
        except Exception as e:
            print("⚠️ Error inserting audit log:", e)

        # -------------------------------
        # FINAL SUCCESS RESPONSE
        # -------------------------------
        return jsonify({
            "report_id": report_id,
            "patient_id": patient_id,
            "primary_diagnosis": diagnostic_result.primary_diagnosis,
            "diagnosis_confidence": diagnostic_result.diagnosis_confidence,
            "secondary_diagnoses": diagnostic_result.secondary_diagnoses,
            "disease_probabilities": diagnostic_result.disease_probabilities,
            "key_findings": diagnostic_result.key_findings,
            "recommendations": diagnostic_result.recommendations,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print("🔥 CRITICAL ERROR:", e, traceback.format_exc())

        return jsonify({
            "report_id": f"error_{int(datetime.utcnow().timestamp())}",
            "patient_id": request.json.get("patient_id"),
            "primary_diagnosis": "Pending Medical Evaluation",
            "diagnosis_confidence": 0.5,
            "secondary_diagnoses": [],
            "disease_probabilities": {"pending": 1.0},
            "key_findings": ["Requires additional clinical evaluation"],
            "recommendations": ["Consult your healthcare provider"],
            "timestamp": datetime.utcnow().isoformat()
        }), 200

OASIS_FEATURES = ("SES", "Age", "eTIV", "MMSE", "nWBV", "ASF", "Sex")
OASIS_MODEL_DIR = Path(__file__).resolve().parent / "ml" / "Models"


@lru_cache(maxsize=1)
def load_oasis_artifacts():
    scaler = joblib.load(OASIS_MODEL_DIR / "dementia_scaler.pkl")

    saved_order = tuple(getattr(scaler, "feature_names_in_", ()))
    if saved_order != OASIS_FEATURES:
        raise ValueError(f"Unexpected OASIS scaler feature order: {saved_order}")

    model = xgb.Booster()
    model.load_model(str(OASIS_MODEL_DIR / "dementia_oasis_model.json"))

    if model.num_features() != len(OASIS_FEATURES):
        raise ValueError("The OASIS model and scaler have different feature counts.")

    return scaler, model


@app.route("/doctor/oasis", methods=["GET", "POST"])
def doctor_oasis():
    if not session.get("doctor_id"):
        return redirect(url_for("doctor_login"))

    values = {}
    score = None
    error = None
    status = 200

    if request.method == "POST":
        values = {
            feature: request.form.get(feature, "").strip()
            for feature in OASIS_FEATURES
        }

        try:
            inputs = {feature: float(values[feature]) for feature in OASIS_FEATURES}

            if not all(isfinite(value) for value in inputs.values()):
                raise ValueError("All inputs must be finite numbers.")
            if inputs["Sex"] not in (0, 1):
                raise ValueError("Select a training sex code.")
            if not 0 <= inputs["MMSE"] <= 30:
                raise ValueError("MMSE must be between 0 and 30.")

        except ValueError:
            error = "Enter valid values for all seven fields. MMSE must be between 0 and 30."
            status = 400
        else:
            try:
                scaler, model = load_oasis_artifacts()

                row = pd.DataFrame([inputs], columns=list(OASIS_FEATURES))
                scaled_row = scaler.transform(row)
                predictions = model.predict(xgb.DMatrix(scaled_row))

                if predictions.shape != (1,):
                    raise ValueError("Expected one OASIS model score.")

                score = float(predictions[0])
                if not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("The OASIS model returned an invalid score.")

            except Exception:
                app.logger.exception("OASIS model inference failed")
                score = None
                error = "The OASIS model could not run. Check the Flask terminal."
                status = 500

    return render_template(
        "doctor_oasis.html",
        features=OASIS_FEATURES,
        values=values,
        score=score,
        error=error,
    ), status        
# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run()
