from datetime import timezone, timedelta

def convert_to_malaysia_time(timestamp):
    """
    Converts a UTC timestamp to Malaysia Time (UTC+8) and returns a formatted string.
    """
    if timestamp:
        if hasattr(timestamp, "to_datetime"):  # Firestore timestamp
            timestamp = timestamp.to_datetime()
        malaysia_timezone = timezone(timedelta(hours=8))
        local_time = timestamp.astimezone(malaysia_timezone)
        return local_time.strftime('%Y-%m-%d %H:%M') + " 🇲🇾"
    else:
        return "Unknown"
