"""Attendance percentages and Excel / PDF exports.

A class counts as held on a day when at least one student was marked Present
or Late for that subject; everyone expected but not marked that day is Absent.
Late arrivals count as attended.
"""
import io
import os
from datetime import datetime

from common import LOW_ATTENDANCE_PERCENT, STATUS_ABSENT, STATUS_LATE, STATUS_PRESENT, roster

SUMMARY_HEADERS = ['Register No', 'Name', 'Subject', 'Classes Held', 'Present', 'Late', 'Absent', 'Attendance %']
DETAIL_HEADERS = ['Date', 'Subject', 'Register No', 'Name', 'Status', 'Time']
PDF_DETAIL_LIMIT = 2500


def _date_range(start, end):
    sql, params = '', []
    if start:
        sql += ' AND date >= ?'
        params.append(start)
    if end:
        sql += ' AND date <= ?'
        params.append(end)
    return sql, params


def class_dates(conn, subject, start=None, end=None):
    sql, params = _date_range(start, end)
    return [row['date'] for row in conn.execute(
        'SELECT DISTINCT date FROM attendance WHERE subject = ?' + sql + ' ORDER BY date',
        [subject] + params)]


def _records(conn, subject, start=None, end=None, register_number=None):
    """{register_number: {date: (status, time)}}"""
    sql, params = _date_range(start, end)
    query = 'SELECT register_number, date, status, time FROM attendance WHERE subject = ?' + sql
    args = [subject] + params
    if register_number is not None:
        query += ' AND register_number = ?'
        args.append(register_number)
    records = {}
    for row in conn.execute(query, args):
        records.setdefault(row['register_number'], {})[row['date']] = (row['status'], row['time'])
    return records


def report_students(conn, subject, records, register_number=None):
    """[(register_number, name)]: the subject's roster plus anyone who has records in it."""
    students = {row['register_number']: row['name'] for row in roster(conn, subject)}
    missing = [reg for reg in records if reg not in students]
    if missing:
        placeholders = ','.join('?' * len(missing))
        for row in conn.execute(f'SELECT register_number, name FROM students '
                                f'WHERE register_number IN ({placeholders})', missing):
            students[row['register_number']] = row['name']
    if register_number is not None:
        students = {reg: name for reg, name in students.items() if reg == register_number}
    return sorted(students.items())


def subject_summary(conn, subject, start=None, end=None, register_number=None):
    held = len(class_dates(conn, subject, start, end))
    records = _records(conn, subject, start, end, register_number)
    rows = []
    for reg, name in report_students(conn, subject, records, register_number):
        statuses = [status for status, _ in records.get(reg, {}).values()]
        present = statuses.count(STATUS_PRESENT)
        late = statuses.count(STATUS_LATE)
        percent = round((present + late) * 100.0 / held, 1) if held else None
        rows.append({
            'register_number': reg, 'name': name, 'subject': subject, 'held': held,
            'present': present, 'late': late, 'absent': held - present - late,
            'percent': percent, 'low': percent is not None and percent < LOW_ATTENDANCE_PERCENT,
        })
    return rows


def subject_details(conn, subject, start=None, end=None, register_number=None):
    dates = class_dates(conn, subject, start, end)
    records = _records(conn, subject, start, end, register_number)
    rows = []
    for reg, name in report_students(conn, subject, records, register_number):
        marks = records.get(reg, {})
        for date in dates:
            status, time = marks.get(date, (STATUS_ABSENT, ''))
            rows.append({'date': date, 'subject': subject, 'register_number': reg, 'name': name,
                         'status': status, 'time': time or ''})
    return rows


def build_report(conn, subjects, start=None, end=None, register_number=None):
    """(summary rows, detail rows) for the given subjects."""
    summary, details = [], []
    for subject in subjects:
        summary.extend(subject_summary(conn, subject, start, end, register_number))
        details.extend(subject_details(conn, subject, start, end, register_number))
    details.sort(key=lambda r: (r['date'], r['subject'], r['register_number']))
    return summary, details


def overall(summary):
    held = sum(r['held'] for r in summary)
    attended = sum(r['present'] + r['late'] for r in summary)
    return {'held': held, 'attended': attended, 'late': sum(r['late'] for r in summary),
            'absent': held - attended,
            'percent': round(attended * 100.0 / held, 1) if held else None}


def _summary_values(row):
    return [row['register_number'], row['name'], row['subject'], row['held'], row['present'],
            row['late'], row['absent'], row['percent'] if row['percent'] is not None else '']


def _detail_values(row):
    return [row['date'], row['subject'], row['register_number'], row['name'], row['status'], row['time']]


def describe_filters(start, end):
    if start and end:
        return f'Period: {start} to {end}'
    if start:
        return f'Period: from {start}'
    if end:
        return f'Period: up to {end}'
    return 'Period: all records'


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------

def to_xlsx(summary, details, title, subtitle):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='1E3A8A')
    low_fill = PatternFill('solid', fgColor='FEE2E2')

    def append(sheet, values):
        sheet.append(values)
        for cell in sheet[sheet.max_row]:
            # Never let a name such as "=HYPERLINK(...)" be stored as a formula.
            if cell.data_type == 'f':
                cell.data_type = 's'
        return sheet[sheet.max_row]

    def add_table(sheet, headers, rows, low_flags=None):
        for cell in append(sheet, headers):
            cell.font, cell.fill = header_font, header_fill
        first = sheet.max_row + 1
        for i, values in enumerate(rows):
            cells = append(sheet, values)
            if low_flags and low_flags[i]:
                for cell in cells:
                    cell.fill = low_fill
        sheet.freeze_panes = sheet.cell(row=first, column=1)
        for column in sheet.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in column[3:] or column)
            sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 10), 45)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Summary'
    append(sheet, [title])[0].font = Font(bold=True, size=14)
    append(sheet, [subtitle])
    append(sheet, [f'Generated {datetime.now():%Y-%m-%d %H:%M}  |  below {LOW_ATTENDANCE_PERCENT}% highlighted'])
    add_table(sheet, SUMMARY_HEADERS, [_summary_values(r) for r in summary], [r['low'] for r in summary])

    records = workbook.create_sheet('Records')
    append(records, [title])[0].font = Font(bold=True, size=14)
    append(records, [subtitle])
    append(records, ['One row per student per class day'])
    add_table(records, DETAIL_HEADERS, [_detail_values(r) for r in details])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def _font_candidates():
    windows_fonts = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
    return [
        (os.path.join(windows_fonts, 'arial.ttf'), os.path.join(windows_fonts, 'arialbd.ttf')),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
    ]


def _setup_font(pdf):
    """Use a Unicode TTF font when one is installed; otherwise Helvetica (Latin-1 only)."""
    for regular, bold in _font_candidates():
        if os.path.isfile(regular) and os.path.isfile(bold):
            pdf.add_font('Report', '', regular)
            pdf.add_font('Report', 'B', bold)
            return 'Report', str

    def latin1(text):
        return str(text).replace('\u2014', '-').encode('latin-1', 'replace').decode('latin-1')
    return 'Helvetica', latin1


def to_pdf(summary, details, title, subtitle):
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    pdf = FPDF(orientation='L', format='A4')
    pdf.set_auto_page_break(True, margin=12)
    family, text = _setup_font(pdf)
    heading_style = FontFace(emphasis='BOLD', fill_color=(226, 232, 240))
    low_style = FontFace(color=(185, 28, 28))

    def heading(value, size):
        pdf.set_font(family, 'B', size)
        pdf.cell(0, 9, text(value), new_x='LMARGIN', new_y='NEXT')

    def line(value):
        pdf.set_font(family, '', 9)
        pdf.cell(0, 5, text(value), new_x='LMARGIN', new_y='NEXT')

    def table(headers, rows, widths, low_flags=None):
        pdf.set_font(family, '', 8)
        with pdf.table(col_widths=widths, first_row_as_headings=True, headings_style=heading_style,
                       line_height=5) as tbl:
            header_row = tbl.row()
            for value in headers:
                header_row.cell(text(value))
            for i, values in enumerate(rows):
                row = tbl.row()
                style = low_style if low_flags and low_flags[i] else None
                for value in values:
                    row.cell(text('' if value is None else value), style=style)

    pdf.add_page()
    heading(title, 15)
    line(subtitle)
    line(f'Generated {datetime.now():%Y-%m-%d %H:%M}  |  below {LOW_ATTENDANCE_PERCENT}% shown in red')
    pdf.ln(3)
    if summary:
        table(SUMMARY_HEADERS, [_summary_values(r) for r in summary],
              (28, 60, 45, 24, 20, 18, 20, 26), [r['low'] for r in summary])
    else:
        line('No students or attendance records for the selected filters.')

    if details:
        pdf.add_page()
        heading('Attendance records', 13)
        shown = details[:PDF_DETAIL_LIMIT]
        if len(details) > PDF_DETAIL_LIMIT:
            line(f'Showing the first {PDF_DETAIL_LIMIT} of {len(details)} rows - '
                 'use the Excel export for the full list.')
        table(DETAIL_HEADERS, [_detail_values(r) for r in shown], (26, 50, 32, 75, 28, 30))
    return bytes(pdf.output())
