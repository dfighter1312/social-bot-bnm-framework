import datetime


def convert_long_date(str):
    """Convert Long Java string to Datetime format."""
    try:
        f = float(str[:-1]) / 1000.0
        dt_format = datetime.datetime.fromtimestamp(f)
        return dt_format
    except:
        return str