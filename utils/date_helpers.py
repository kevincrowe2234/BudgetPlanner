from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from utils.config import DATE_FORMATS

def parse_date_string(date_str):
    """Parse date string using multiple formats"""
    try:
        return parse(date_str, fuzzy=True)
    except ValueError:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    return None

def format_date(date_obj, fmt="%Y-%m-%d"):
    """Format date object as string"""
    if isinstance(date_obj, datetime):
        return date_obj.strftime(fmt)
    return str(date_obj)

def generate_date_range(start_date, months=12):
    """Generate a date range for projections"""
    end_date = start_date + relativedelta(months=months)
    dates = []
    current = start_date
    
    while current <= end_date:
        dates.append(current)
        current = current + timedelta(days=1)
        
    return dates

def calculate_next_occurrence(date, frequency):
    """Calculate next occurrence of a recurring transaction"""
    if frequency == "Daily":
        return date + timedelta(days=1)
    elif frequency == "Weekly":
        return date + timedelta(weeks=1)
    elif frequency == "Biweekly":
        return date + timedelta(weeks=2)
    elif frequency == "Monthly":
        return date + relativedelta(months=1)
    elif frequency == "Quarterly":
        return date + relativedelta(months=3)
    elif frequency == "Annually":
        return date + relativedelta(years=1)
    else:
        return None  # Not recurring