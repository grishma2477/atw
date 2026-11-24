Create database and user:
```sql
CREATE DATABASE neurosense;

CREATE USER 'neurosense'@'localhost' IDENTIFIED BY 'neurosense@123';

GRANT ALL PRIVILEGES ON neurosense.* TO 'neurosense'@'localhost';

FLUSH PRIVILEGES;
```

Create required tables:
```sql
USE neurosense;

CREATE TABLE patients (
    id INT PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    password VARCHAR(255),
    phone VARCHAR(20),
    age INT,
    gender ENUM('Male', 'Female', 'Other'),
    blood_type VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE doctors (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    password VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE assessments (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    form_data LONGTEXT,
    alz LONGTEXT,
    park LONGTEXT,
    dem LONGTEXT,
    diag LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE diagnostic_reports (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    doctor_id INT,
    primary_diagnosis VARCHAR(255),
    diagnosis_confidence FLOAT,
    secondary_diagnoses LONGTEXT,
    disease_probabilities LONGTEXT,
    key_findings LONGTEXT,
    recommendations LONGTEXT,
    mmse_score INT,
    cdr_score FLOAT,
    csf_tau_level FLOAT,
    csf_abeta42_level FLOAT,
    apoe_status VARCHAR(10),
    mri_file_path VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id),
    FOREIGN KEY (doctor_id) REFERENCES doctors(id)
);

CREATE TABLE health_events (
    id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    event_type ENUM('diagnosis', 'intervention', 'medication', 'assessment', 'screening'),
    title VARCHAR(255),
    description LONGTEXT,
    severity ENUM('low', 'moderate', 'high'),
    disease VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS doctor_patient_access (
    doctor_id BIGINT UNSIGNED NOT NULL,
    patient_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (doctor_id, patient_id)
);

CREATE TABLE IF NOT EXISTS screening_reports (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    patient_id BIGINT UNSIGNED NOT NULL,
    doctor_id BIGINT UNSIGNED NOT NULL,
    result_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX screening_patient_history (patient_id, created_at)
);


CREATE TABLE audit_log (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    action VARCHAR(50),
    table_name VARCHAR(100),
    record_id INT,
    changes LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```