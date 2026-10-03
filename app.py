"""Web dashboard for the face recognition attendance system.

Three roles: admin (manages everything), staff (one login per subject) and
students (view their own attendance). The camera tools run as separate
processes started from here.
"""
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import wraps

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.utils import secure_filename

import common
import reports
from common import (DATE_FORMAT, DEFAULT_ADMIN_PASSWORD, GENERAL_SUBJECT, LOW_ATTENDANCE_PERCENT,
                    MATCH_THRESHOLD, MIN_PASSWORD_LENGTH, PHOTOS_PER_STUDENT, STATUS_ABSENT,
                    STATUS_LATE, STATUS_PRESENT, connect)

STATUSES = (STATUS_PRESENT, STATUS_LATE, STATUS_ABSENT)
HOME = {'admin': 'admin_dashboard', 'staff': 'staff_dashboard', 'student': 'student_dashboard'}
NAV_LINKS = {
    'admin': [('admin_dashboard', 'Dashboard'), ('admin_students', 'Students'),
              ('admin_subjects', 'Subjects'), ('admin_reports', 'Reports'),
              ('admin_logs', 'Change Log'), ('admin_accuracy', 'Accuracy')],
    'staff': [('staff_dashboard', 'Dashboard'), ('staff_reports', 'Reports')],
    'student': [('student_dashboard', 'My Attendance')],
}


def load_secret_key():
    """SECRET_KEY from the environment, else a random key created once in data/.secret_key.

    The key signs the login cookie; a hardcoded one would let anyone forge an
    admin session.
    """
    key = os.environ.get('SECRET_KEY')
    if key:
        return key
    try:
        with open(common.SECRET_KEY_FILE, encoding='utf-8') as f:
            key = f.read().strip()
    except FileNotFoundError:
        key = ''
    if not key:
        key = secrets.token_hex(32)
        os.makedirs(common.DATA_DIR, exist_ok=True)
        with open(common.SECRET_KEY_FILE, 'w', encoding='utf-8') as f:
            f.write(key)
    return key


app = Flask(__name__)
app.secret_key = load_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,   # bulk-import CSV uploads
)

# Run database initialization / migrations
common.init_db()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def wants_json():
    return request.accept_mimetypes.best_match(['application/json', 'text/html']) == 'application/json'


def today():
    return datetime.now().strftime(DATE_FORMAT)


def parse_date(value):
    """A 'YYYY-MM-DD' value from a form or query string, or '' if missing/invalid."""
    value = (value or '').strip()
    try:
        datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        return ''
    return value


def parse_minutes(value, default=0.0, maximum=600.0):
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return default
    if minutes != minutes:      # NaN
        return default
    return min(max(minutes, 0.0), maximum)


def actor():
    """(role, display name) of the logged-in user, for the audit log."""
    role = session.get('role')
    if role == 'staff':
        professor = session.get('professor_name')
        return role, f"{professor} ({session['subject']})" if professor else session['subject']
    return role, session.get('username', '')


def load_evaluation():
    try:
        with open(common.EVALUATION_JSON, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def evaluation_row(data, threshold=MATCH_THRESHOLD):
    """The evaluation result for the threshold currently in use."""
    for row in (data or {}).get('thresholds', []):
        if abs(row['threshold'] - threshold) < 1e-9:
            return row
    return None


def count_statuses(rows):
    counts = dict.fromkeys(STATUSES, 0)
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    return counts


def attendance_rows(conn, subject, date):
    """Every expected student for a subject and day, with their status."""
    marks = {row['register_number']: row for row in conn.execute(
        'SELECT register_number, status, time FROM attendance WHERE subject = ? AND date = ?',
        (subject, date))}
    rows = []
    for reg, name in reports.report_students(conn, subject, marks):
        mark = marks.get(reg)
        rows.append({'reg_no': reg, 'name': name,
                     'status': mark['status'] if mark else STATUS_ABSENT,
                     'time': mark['time'] if mark else '—'})
    return rows


def recent_sessions(conn, subject=None, limit=10):
    sql = '''
        SELECT s.*,
            (SELECT COUNT(*) FROM attendance a WHERE a.subject = s.subject AND a.date = s.date
                AND a.status = 'Present') AS present_count,
            (SELECT COUNT(*) FROM attendance a WHERE a.subject = s.subject AND a.date = s.date
                AND a.status = 'Late') AS late_count
        FROM sessions s'''
    args = []
    if subject is not None:
        sql += ' WHERE s.subject = ?'
        args.append(subject)
    sql += ' ORDER BY s.started_at DESC, s.id DESC LIMIT ?'
    args.append(limit)
    return conn.execute(sql, args).fetchall()


def change_attendance(conn, register_number, subject, date, status):
    """Validate and apply a manual Present/Late/Absent change. Returns an error message or None."""
    if status not in STATUSES:
        return 'Invalid attendance status.'
    if status != STATUS_ABSENT and not common.on_roster(conn, register_number, subject):
        return 'That student is not assigned to this subject.'
    if common.set_attendance_status(conn, register_number, subject, date, status, *actor()) is None:
        return 'Student not found.'
    return None


# --------------------------------------------------------------------------
# Security: CSRF tokens, deleted accounts, response headers
# --------------------------------------------------------------------------

def csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = session['_csrf_token'] = secrets.token_urlsafe(32)
    return token


app.jinja_env.globals['csrf_token'] = csrf_token


@app.before_request
def protect():
    if request.method == 'POST':
        sent = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token') or ''
        expected = session.get('_csrf_token') or ''
        if not expected or not secrets.compare_digest(sent, expected):
            message = 'Your session expired - please reload the page and try again.'
            if wants_json():
                return jsonify(status='error', message=message), 400
            flash(message, 'danger')
            return redirect(url_for('login_page'))

    # A staff or student account deleted while logged in must stop working.
    role = session.get('role')
    if role in ('staff', 'student') and request.endpoint != 'logout':
        conn = connect()
        try:
            if role == 'staff':
                exists = conn.execute('SELECT 1 FROM staff WHERE subject_name = ?',
                                      (session.get('subject'),)).fetchone()
            else:
                exists = conn.execute('SELECT 1 FROM students WHERE register_number = ?',
                                      (session.get('reg_no'),)).fetchone()
        finally:
            conn.close()
        if not exists:
            session.clear()
            flash('Your account no longer exists. Please contact the administrator.', 'danger')
            return redirect(url_for('login_page'))
    return None


@app.after_request
def security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    return response


@app.errorhandler(413)
def file_too_large(_error):
    flash('That file is too large (max 2 MB).', 'danger')
    return redirect(url_for('admin_students'))


def role_required(*roles):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get('role') not in roles:
                if wants_json():
                    return jsonify(status='error', message='Unauthorized - please log in again.'), 401
                return redirect(url_for('login_page'))
            return view(*args, **kwargs)
        return wrapped
    return decorator


@app.context_processor
def inject_globals():
    return {'nav_links': NAV_LINKS.get(session.get('role'), []),
            'low_percent': LOW_ATTENDANCE_PERCENT,
            'general_subject': GENERAL_SUBJECT,
            'min_password': MIN_PASSWORD_LENGTH,
            'photos_target': PHOTOS_PER_STUDENT}


@app.template_filter('ts')
def format_timestamp(value):
    try:
        return datetime.strptime(value, common.TIMESTAMP_FORMAT).strftime('%d %b %Y, %I:%M %p')
    except (TypeError, ValueError):
        return value or '—'


@app.template_filter('pct')
def format_percent(value):
    return '—' if value is None else f'{value:.1f}%'


# --------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------

@app.route('/')
def login_page():
    role = session.get('role')
    if role in HOME:
        return redirect(url_for(HOME[role]))
    return render_template('login.html')


def start_user_session(role, **values):
    session.clear()     # fresh session (and CSRF token) on every login
    session['role'] = role
    session.update(values)


def upgrade_password_hash(conn, table, key, row, password):
    if not common.is_password_hash(row['password']):
        conn.execute(f'UPDATE {table} SET password = ? WHERE {key} = ?',
                     (common.hash_password(password), row[key]))
        conn.commit()


@app.route('/login', methods=['POST'])
def login():
    role = request.form.get('role')
    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''

    conn = connect()
    try:
        if role == 'admin':
            row = conn.execute('SELECT * FROM admins WHERE username = ?', (username,)).fetchone()
            if row and common.verify_password(row['password'], password):
                upgrade_password_hash(conn, 'admins', 'username', row, password)
                start_user_session('admin', username=row['username'], display_name='Admin',
                                   default_password=(password == DEFAULT_ADMIN_PASSWORD))
                return redirect(url_for('admin_dashboard'))
            flash('Invalid admin credentials.', 'danger')
        elif role == 'staff':
            row = conn.execute('SELECT * FROM staff WHERE subject_name = ?', (username,)).fetchone()
            if row and common.verify_password(row['password'], password):
                upgrade_password_hash(conn, 'staff', 'subject_name', row, password)
                professor = row['professor_name'] or ''
                start_user_session('staff', username=row['subject_name'], subject=row['subject_name'],
                                   professor_name=professor,
                                   display_name=f"{professor} · {row['subject_name']}" if professor
                                   else row['subject_name'])
                return redirect(url_for('staff_dashboard'))
            flash('Invalid staff credentials.', 'danger')
        elif role == 'student':
            # Register number first; the name works too when it is unique enough.
            rows = (conn.execute('SELECT * FROM students WHERE register_number = ?', (username,)).fetchall()
                    or conn.execute('SELECT * FROM students WHERE name = ?', (username,)).fetchall())
            row = next((r for r in rows if common.verify_password(r['password'], password)), None)
            if row:
                upgrade_password_hash(conn, 'students', 'register_number', row, password)
                start_user_session('student', reg_no=row['register_number'], name=row['name'],
                                   display_name=f"{row['name']} ({row['register_number']})")
                return redirect(url_for('student_dashboard'))
            flash('Invalid student credentials.', 'danger')
        else:
            # Unknown or missing role: never fall through returning None (Flask 500).
            flash('Please select a valid role.', 'danger')
    finally:
        conn.close()
    return redirect(url_for('login_page'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))


def account_ref():
    role = session['role']
    if role == 'admin':
        return 'admins', 'username', session['username']
    if role == 'staff':
        return 'staff', 'subject_name', session['subject']
    return 'students', 'register_number', session['reg_no']


@app.route('/change-password', methods=['GET', 'POST'])
@role_required('admin', 'staff', 'student')
def change_password():
    if request.method == 'POST':
        current = request.form.get('current_password') or ''
        new = request.form.get('new_password') or ''
        confirm = request.form.get('confirm_password') or ''
        table, key, value = account_ref()
        conn = connect()
        try:
            row = conn.execute(f'SELECT password FROM {table} WHERE {key} = ?', (value,)).fetchone()
            if not row or not common.verify_password(row['password'], current):
                error = 'Your current password is incorrect.'
            elif new != confirm:
                error = 'The new passwords do not match.'
            elif new == current:
                error = 'Choose a new password that is different from the current one.'
            else:
                error = common.validate_password(new)
            if error:
                flash(error, 'danger')
                return redirect(url_for('change_password'))
            conn.execute(f'UPDATE {table} SET password = ? WHERE {key} = ?', (common.hash_password(new), value))
            conn.commit()
        finally:
            conn.close()
        session.pop('default_password', None)
        flash('Your password has been changed.', 'success')
        return redirect(url_for(HOME[session['role']]))
    return render_template('change_password.html')


# --------------------------------------------------------------------------
# Admin
# --------------------------------------------------------------------------

@app.route('/admin-dashboard')
@role_required('admin')
def admin_dashboard():
    conn = connect()
    try:
        staffs = conn.execute('SELECT subject_name, professor_name FROM staff '
                              'ORDER BY subject_name COLLATE NOCASE').fetchall()
        subjects = [GENERAL_SUBJECT] + [s['subject_name'] for s in staffs]
        subject = request.args.get('subject') or GENERAL_SUBJECT
        if subject not in subjects:
            subject = GENERAL_SUBJECT
        date = parse_date(request.args.get('date')) or today()
        rows = attendance_rows(conn, subject, date)
        sessions_list = recent_sessions(conn)
        open_to_all = subject != GENERAL_SUBJECT and not common.has_enrollments(conn, subject)
    finally:
        conn.close()
    evaluation = load_evaluation()
    return render_template('admin_dashboard.html', staffs=staffs, subject=subject, date=date, rows=rows,
                           counts=count_statuses(rows), open_to_all=open_to_all, sessions=sessions_list,
                           evaluation=evaluation, evaluation_row=evaluation_row(evaluation),
                           threshold=MATCH_THRESHOLD, models_missing=common.missing_model_files(),
                           photos_target=PHOTOS_PER_STUDENT)


@app.route('/admin/attendance', methods=['POST'])
@role_required('admin')
def admin_set_attendance():
    subject = request.form.get('subject') or ''
    date = parse_date(request.form.get('date'))
    conn = connect()
    try:
        if not date or not common.is_known_subject(conn, subject):
            error = 'Invalid attendance change.'
        else:
            error = change_attendance(conn, request.form.get('register_number') or '', subject, date,
                                      request.form.get('status'))
    finally:
        conn.close()
    if error:
        flash(error, 'danger')
    return redirect(url_for('admin_dashboard', subject=subject or None, date=date or None))


@app.route('/admin/students')
@role_required('admin')
def admin_students():
    conn = connect()
    try:
        students = conn.execute('SELECT register_number, name FROM students ORDER BY register_number').fetchall()
        enrolled = {}
        for row in conn.execute('SELECT register_number, subject_name FROM enrollments '
                                'ORDER BY subject_name COLLATE NOCASE'):
            enrolled.setdefault(row['register_number'], []).append(row['subject_name'])
    finally:
        conn.close()
    return render_template('admin_students.html', students=students, enrolled=enrolled,
                           photo_counts=common.face_photo_counts(), photos_target=PHOTOS_PER_STUDENT)


@app.route('/manage-student', methods=['POST'])
@role_required('admin')
def manage_student():
    action = request.form.get('action')
    reg = (request.form.get('register_number') or '').strip()
    name = (request.form.get('name') or '').strip()
    password = request.form.get('password') or ''

    conn = connect()
    try:
        row = conn.execute('SELECT name FROM students WHERE register_number = ?', (reg,)).fetchone()
        if action == 'add':
            error = (common.validate_register_number(reg) or common.validate_name(name)
                     or common.validate_password(password)
                     or ('A student with this register number already exists.' if row else None))
            if error:
                flash(error, 'danger')
            else:
                conn.execute('INSERT INTO students (register_number, name, password) VALUES (?, ?, ?)',
                             (reg, name, common.hash_password(password)))
                conn.commit()
                flash(f'Student {name} ({reg}) added.', 'success')
        elif action == 'edit':
            error = ((None if row else 'Student not found.') or common.validate_name(name)
                     or (common.validate_password(password) if password else None))
            if error:
                flash(error, 'danger')
            else:
                if password:
                    conn.execute('UPDATE students SET name = ?, password = ? WHERE register_number = ?',
                                 (name, common.hash_password(password), reg))
                else:
                    conn.execute('UPDATE students SET name = ? WHERE register_number = ?', (name, reg))
                conn.execute('UPDATE attendance SET name = ? WHERE register_number = ?', (name, reg))
                conn.commit()
                flash(f'Student {name} ({reg}) updated.', 'success')
        elif action == 'delete':
            if not row:
                flash('Student not found.', 'danger')
            else:
                removed = conn.execute('DELETE FROM attendance WHERE register_number = ?', (reg,)).rowcount
                conn.execute('DELETE FROM enrollments WHERE register_number = ?', (reg,))
                conn.execute('DELETE FROM students WHERE register_number = ?', (reg,))
                common.log_action(conn, *actor(), 'Deleted student', register_number=reg,
                                  student_name=row['name'], details=f'{removed} attendance record(s) removed')
                conn.commit()
                folders = common.delete_face_folders(reg)
                if folders:
                    refresh_features()
                flash(f"Student {row['name']} ({reg}) deleted with {removed} attendance record(s)"
                      + (f" and {folders} face photo folder(s)." if folders else "."), 'success')
        else:
            flash('Unknown action.', 'danger')
    finally:
        conn.close()
    return redirect(url_for('admin_students'))


@app.route('/admin/students/import', methods=['POST'])
@role_required('admin')
def import_students():
    upload = request.files.get('csv_file')
    if not upload or not upload.filename:
        flash('Choose a CSV file to import.', 'warning')
        return redirect(url_for('admin_students'))
    raw = upload.read()
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    conn = connect()
    try:
        result = common.import_students_csv(conn, text)
    finally:
        conn.close()

    summary = (f"Import finished: {result['added']} added, {result['updated']} updated, "
               f"{result['enrolled']} subject assignment(s).")
    errors = result['errors']
    if errors:
        shown = '\n'.join(errors[:10]) + (f'\n... and {len(errors) - 10} more' if len(errors) > 10 else '')
        flash(f"{summary}\n{len(errors)} problem(s):\n{shown}",
              'warning' if result['added'] or result['updated'] else 'danger')
    else:
        flash(summary, 'success')
    return redirect(url_for('admin_students'))


@app.route('/admin/students/sample.csv')
@role_required('admin')
def sample_students_csv():
    sample = ('register_number,name,password,subjects\r\n'
              '21CS001,Arun Kumar,changeme1,Maths;Physics\r\n'
              '21CS002,Priya S,changeme2,\r\n')
    return Response(sample, mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="students_sample.csv"'})


@app.route('/admin/subjects')
@role_required('admin')
def admin_subjects():
    conn = connect()
    try:
        staffs = conn.execute('SELECT subject_name, professor_name FROM staff '
                              'ORDER BY subject_name COLLATE NOCASE').fetchall()
        students = conn.execute('SELECT register_number, name FROM students ORDER BY register_number').fetchall()
        enrollment = {s['subject_name']: [] for s in staffs}
        for row in conn.execute('SELECT subject_name, register_number FROM enrollments ORDER BY register_number'):
            if row['subject_name'] in enrollment:
                enrollment[row['subject_name']].append(row['register_number'])
        record_counts = {row['subject']: row['n'] for row in conn.execute(
            'SELECT subject, COUNT(*) AS n FROM attendance GROUP BY subject')}
    finally:
        conn.close()
    enroll_data = {'students': [[s['register_number'], s['name']] for s in students], 'enrollment': enrollment}
    return render_template('admin_subjects.html', staffs=staffs, student_count=len(students),
                           enrollment=enrollment, enroll_data=enroll_data, record_counts=record_counts)


@app.route('/manage-staff', methods=['POST'])
@role_required('admin')
def manage_staff():
    action = request.form.get('action')
    subject = (request.form.get('subject_name') or '').strip()
    professor = (request.form.get('professor_name') or '').strip()
    password = request.form.get('password') or ''

    conn = connect()
    try:
        exists = conn.execute('SELECT 1 FROM staff WHERE subject_name = ?', (subject,)).fetchone() is not None
        if action == 'add':
            error = (common.validate_subject_name(subject) or common.validate_password(password)
                     or ('This subject already exists.' if exists else None))
            if error:
                flash(error, 'danger')
            else:
                conn.execute('INSERT INTO staff (subject_name, professor_name, password) VALUES (?, ?, ?)',
                             (subject, professor, common.hash_password(password)))
                conn.commit()
                flash(f'Subject {subject} added.', 'success')
        elif action == 'edit':
            error = (None if exists else 'Subject not found.') or (
                common.validate_password(password) if password else None)
            if error:
                flash(error, 'danger')
            else:
                if password:
                    conn.execute('UPDATE staff SET professor_name = ?, password = ? WHERE subject_name = ?',
                                 (professor, common.hash_password(password), subject))
                else:
                    conn.execute('UPDATE staff SET professor_name = ? WHERE subject_name = ?', (professor, subject))
                conn.commit()
                flash(f'Subject {subject} updated.', 'success')
        elif action == 'delete':
            if not exists:
                flash('Subject not found.', 'danger')
            else:
                removed = conn.execute('DELETE FROM attendance WHERE subject = ?', (subject,)).rowcount
                conn.execute('DELETE FROM sessions WHERE subject = ?', (subject,))
                conn.execute('DELETE FROM enrollments WHERE subject_name = ?', (subject,))
                conn.execute('DELETE FROM staff WHERE subject_name = ?', (subject,))
                common.log_action(conn, *actor(), 'Deleted subject', subject=subject,
                                  details=f'{removed} attendance record(s) removed')
                conn.commit()
                flash(f'Subject {subject} deleted with {removed} attendance record(s).', 'success')
        else:
            flash('Unknown action.', 'danger')
    finally:
        conn.close()
    return redirect(url_for('admin_subjects'))


@app.route('/admin/subjects/enrollment', methods=['POST'])
@role_required('admin')
def update_enrollment():
    subject = request.form.get('subject_name') or ''
    chosen = set(request.form.getlist('register_numbers'))
    conn = connect()
    try:
        if not conn.execute('SELECT 1 FROM staff WHERE subject_name = ?', (subject,)).fetchone():
            flash('Subject not found.', 'danger')
        else:
            valid = [row[0] for row in conn.execute('SELECT register_number FROM students ORDER BY register_number')
                     if row[0] in chosen]
            conn.execute('DELETE FROM enrollments WHERE subject_name = ?', (subject,))
            conn.executemany('INSERT INTO enrollments (register_number, subject_name) VALUES (?, ?)',
                             [(reg, subject) for reg in valid])
            conn.commit()
            flash(f'{subject}: {len(valid)} student(s) assigned.' if valid else
                  f'{subject}: no students assigned, so every student is expected in this subject.', 'success')
    finally:
        conn.close()
    return redirect(url_for('admin_subjects'))


def report_filters():
    return parse_date(request.args.get('start')), parse_date(request.args.get('end'))


def export_args(**values):
    return {key: value for key, value in values.items() if value}


@app.route('/admin/reports')
@role_required('admin')
def admin_reports():
    start, end = report_filters()
    conn = connect()
    try:
        options = common.all_subjects(conn)
        subject = request.args.get('subject') or 'ALL'
        if subject != 'ALL' and not common.is_known_subject(conn, subject):
            subject = 'ALL'
        chosen = options if subject == 'ALL' else [subject]
        rows = []
        for name in chosen:
            rows.extend(reports.subject_summary(conn, name, start, end))
    finally:
        conn.close()
    return render_template('reports.html', rows=rows, subject=subject, subject_options=options,
                           start=start, end=end, endpoint='admin_reports',
                           export=export_args(subject=None if subject == 'ALL' else subject, start=start, end=end))


@app.route('/admin/logs')
@role_required('admin')
def admin_logs():
    conn = connect()
    try:
        entries = conn.execute('SELECT * FROM audit_log ORDER BY id DESC LIMIT 500').fetchall()
    finally:
        conn.close()
    return render_template('admin_logs.html', entries=entries)


@app.route('/admin/accuracy')
@role_required('admin')
def admin_accuracy():
    return render_template('admin_accuracy.html', evaluation=load_evaluation(), threshold=MATCH_THRESHOLD,
                           running=is_running('evaluate'), models_missing=common.missing_model_files())


# --------------------------------------------------------------------------
# Staff
# --------------------------------------------------------------------------

@app.route('/staff-dashboard')
@role_required('staff')
def staff_dashboard():
    subject = session['subject']
    date = parse_date(request.args.get('date')) or today()
    conn = connect()
    try:
        row = conn.execute('SELECT professor_name FROM staff WHERE subject_name = ?', (subject,)).fetchone()
        professor_name = row['professor_name'] if row and row['professor_name'] else ''
        rows = attendance_rows(conn, subject, date)
        sessions_list = recent_sessions(conn, subject)
        open_to_all = not common.has_enrollments(conn, subject)
    finally:
        conn.close()
    return render_template('staff.html', subject=subject, professor_name=professor_name, date=date,
                           is_today=(date == today()), rows=rows, counts=count_statuses(rows),
                           sessions=sessions_list, open_to_all=open_to_all,
                           models_missing=common.missing_model_files())


@app.route('/staff/attendance', methods=['POST'])
@role_required('staff')
def staff_set_attendance():
    subject = session['subject']
    date = parse_date(request.form.get('date'))
    if date != today():
        flash('Attendance can only be changed for today.', 'warning')
    else:
        conn = connect()
        try:
            error = change_attendance(conn, request.form.get('register_number') or '', subject, date,
                                      request.form.get('status'))
        finally:
            conn.close()
        if error:
            flash(error, 'danger')
    return redirect(url_for('staff_dashboard', date=date or None))


@app.route('/staff/reports')
@role_required('staff')
def staff_reports():
    subject = session['subject']
    start, end = report_filters()
    conn = connect()
    try:
        rows = reports.subject_summary(conn, subject, start, end)
    finally:
        conn.close()
    return render_template('reports.html', rows=rows, subject=subject, subject_options=None,
                           start=start, end=end, endpoint='staff_reports',
                           export=export_args(start=start, end=end))


# --------------------------------------------------------------------------
# Student
# --------------------------------------------------------------------------

@app.route('/student-dashboard')
@role_required('student')
def student_dashboard():
    reg = session['reg_no']
    start, end = report_filters()
    chosen = request.args.get('subject') or ''
    conn = connect()
    try:
        subjects = common.subjects_for_student(conn, reg)
        if chosen not in subjects:
            chosen = ''
        professors = {row['subject_name']: row['professor_name'] for row in conn.execute(
            'SELECT subject_name, professor_name FROM staff')}
        summary, history = reports.build_report(conn, [chosen] if chosen else subjects, start, end, reg)
    finally:
        conn.close()
    history.sort(key=lambda r: (r['date'], r['subject']), reverse=True)
    return render_template('student.html', student_name=session['name'], reg_no=reg, subjects=subjects,
                           chosen=chosen, start=start, end=end, summary=summary,
                           overall=reports.overall(summary), history=history, professors=professors,
                           export=export_args(subject=chosen, start=start, end=end))


# --------------------------------------------------------------------------
# Excel / PDF export (scoped to what each role may see)
# --------------------------------------------------------------------------

@app.route('/export/<fmt>')
@role_required('admin', 'staff', 'student')
def export(fmt):
    if fmt not in ('xlsx', 'pdf'):
        abort(404)
    start, end = report_filters()
    requested = request.args.get('subject') or ''
    role = session['role']
    register_number = None
    conn = connect()
    try:
        if role == 'staff':
            subjects = [session['subject']]
            title = f"{session['subject']} - attendance report"
        elif role == 'student':
            register_number = session['reg_no']
            subjects = common.subjects_for_student(conn, register_number)
            if requested:
                subjects = [s for s in subjects if s == requested]
            title = f"Attendance - {session['name']} ({register_number})"
        elif requested:
            subjects = [requested] if common.is_known_subject(conn, requested) else []
            title = f'{requested} - attendance report'
        else:
            subjects = common.all_subjects(conn)
            title = 'Attendance report - all subjects'
        summary, details = reports.build_report(conn, subjects, start, end, register_number)
    finally:
        conn.close()

    subtitle = reports.describe_filters(start, end)
    stem = secure_filename(title.replace(' - ', '_')) or 'attendance'
    filename = f"{stem}_{datetime.now():%Y%m%d}.{fmt}"
    if fmt == 'xlsx':
        data = reports.to_xlsx(summary, details, title, subtitle)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        data = reports.to_pdf(summary, details, title, subtitle)
        mimetype = 'application/pdf'
    return Response(data, mimetype=mimetype,
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


# --------------------------------------------------------------------------
# Camera tools and background jobs (separate processes)
# --------------------------------------------------------------------------

_processes = {}
_process_info = {}
_process_lock = threading.Lock()


def is_running(name):
    proc = _processes.get(name)
    return proc is not None and proc.poll() is None


def start_process(name, *args):
    _processes[name] = subprocess.Popen([sys.executable, *args], cwd=common.BASE_DIR)


def camera_busy_message():
    # There is only one webcam: registration and scanning cannot run together.
    if is_running('register'):
        return 'The face registration window is still open - close it first.'
    if is_running('scanner'):
        return 'An attendance scanner is already running - close it first.'
    return None


def models_missing_message():
    missing = common.missing_model_files()
    if missing:
        return 'Face recognition models are missing. Run "python download_models.py" in the project folder.'
    return None


def refresh_features():
    """Rebuild features_all.csv in the background (e.g. after a student's photos were deleted)."""
    with _process_lock:
        if not is_running('extract') and not common.missing_model_files():
            start_process('extract', 'features_extraction_to_csv.py')


@app.route('/run-register', methods=['POST'])
@role_required('admin')
def run_register():
    with _process_lock:
        error = models_missing_message() or camera_busy_message()
        if error:
            return jsonify(status='error', message=error)
        if os.path.exists(common.LATENCY_JSON):
            try:
                os.remove(common.LATENCY_JSON)
            except OSError:
                pass
        start_process('register', 'get_faces_from_camera_tkinter.py')
    return jsonify(status='success', message='Face registration window opened.')


@app.route('/run-extract', methods=['POST'])
@role_required('admin')
def run_extract():
    with _process_lock:
        error = models_missing_message() or (
            'Face processing is already running.' if is_running('extract') else None)
        if error:
            return jsonify(status='error', message=error)
        start_process('extract', 'features_extraction_to_csv.py')
    return jsonify(status='success', message='Feature extraction started in the background.')


@app.route('/run-attendance', methods=['POST'])
@role_required('admin', 'staff')
def run_attendance():
    payload = request.get_json(silent=True) or {}
    if session['role'] == 'staff':
        # A staff account *is* a subject; ignore any client-supplied value.
        subject = session['subject']
    else:
        # Admin has no subject bound to the session, so take it from the request.
        subject = (str(payload.get('subject') or request.form.get('subject') or '')).strip() or GENERAL_SUBJECT
        conn = connect()
        try:
            known = common.is_known_subject(conn, subject)
        finally:
            conn.close()
        if not known:
            return jsonify(status='error', message=f'Unknown subject: {subject}')
    duration = parse_minutes(payload.get('duration'))
    late_after = parse_minutes(payload.get('late_after'))

    with _process_lock:
        error = models_missing_message() or camera_busy_message()
        if error:
            return jsonify(status='error', message=error)
        start_process('scanner', 'attendance_taker.py', f'--duration={duration:g}',
                      f'--late-after={late_after:g}', f'--started-by={actor()[1]}', '--', subject)
        _process_info['scanner'] = subject

    details = []
    if duration:
        details.append(f'stops after {duration:g} min')
    if late_after:
        details.append(f'late after {late_after:g} min')
    return jsonify(status='success', message=f'Attendance scanner opened for {subject}'
                   + (f" ({', '.join(details)})." if details else '.'))


@app.route('/run-evaluation', methods=['POST'])
@role_required('admin')
def run_evaluation():
    with _process_lock:
        error = models_missing_message() or (
            'An accuracy evaluation is already running.' if is_running('evaluate') else None)
        if error:
            return jsonify(status='error', message=error)
        start_process('evaluate', 'evaluate_accuracy.py')
    return jsonify(status='success', message='Accuracy evaluation started - results appear when it finishes.')


@app.route('/api/process-status')
@role_required('admin', 'staff')
def api_process_status():
    status = {name: is_running(name) for name in ('register', 'scanner', 'extract', 'evaluate')}
    status['scanner_subject'] = _process_info.get('scanner') if status['scanner'] else None
    return jsonify(status)


@app.route('/api/registration-latency')
def api_registration_latency():
    # Exposes the reg no/name of the student being enrolled - admin only.
    if session.get('role') != 'admin':
        return jsonify({"registered": False})
    try:
        # Show the graph while registering and for 10 minutes afterwards.
        fresh = is_running('register') or time.time() - os.path.getmtime(common.LATENCY_JSON) < 600
        if fresh:
            with open(common.LATENCY_JSON, encoding='utf-8') as f:
                data = json.load(f)
            if data.get("registered") and data.get("photos_count", 0) > 0:
                data['photos_target'] = PHOTOS_PER_STUDENT
                return jsonify(data)
    except (OSError, ValueError, AttributeError):
        pass
    return jsonify({"registered": False})


# --------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------

def print_startup_summary():
    print("\n" + "=" * 80)
    print("      FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 80)
    if common.missing_model_files():
        print(" ! Face recognition models are missing - run:  python download_models.py")
    data = load_evaluation()
    if not data:
        print(" Accuracy: not measured yet - run  python evaluate_accuracy.py  after registering students.")
    elif data.get('error'):
        print(f" Last accuracy evaluation ({data.get('evaluated_at')}): {data['error']}")
    else:
        ds = data['dataset']
        print(f" Last accuracy evaluation: {data['evaluated_at']} "
              f"({ds['students']} students, {ds['faces_detected']} photos)")
        print("-" * 80)
        print(f" {'Threshold':<20} | {'Correct':<9} | {'False reject':<12} | {'False accept':<12} | {'Accuracy':<9}")
        print("-" * 80)
        for row in data['thresholds']:
            tags = []
            if abs(row['threshold'] - MATCH_THRESHOLD) < 1e-9:
                tags.append('in use')
            if data.get('recommended_threshold') is not None and \
                    abs(row['threshold'] - data['recommended_threshold']) < 1e-9:
                tags.append('best')
            label = f"{row['threshold']:.2f}" + (f" ({', '.join(tags)})" if tags else '')
            print(f" {label:<20} | {format_percent(row['tar']):<9} | {format_percent(row['frr']):<12} | "
                  f"{format_percent(row['far']):<12} | {format_percent(row['accuracy']):<9}")
    print("=" * 80)
    print(f" Open http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '5000')} in your browser\n")


if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":   # not the debug reloader's child process
        if os.path.exists(common.LATENCY_JSON):
            try:
                os.remove(common.LATENCY_JSON)
            except OSError:
                pass
        print_startup_summary()
    app.run(host=os.environ.get('HOST', '127.0.0.1'),
            port=int(os.environ.get('PORT', '5000')),
            debug=os.environ.get('FLASK_DEBUG') == '1')
