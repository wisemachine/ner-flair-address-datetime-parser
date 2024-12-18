import datefinder as df
import datetime
from dateutil import parser as date_parser
import warnings
import re
import json

LEN_PARSED_DATE_THRESHOLD = 2


class DateTimeParser:
    """A class for parsing, formatting, and manipulating date and time information."""

    # Function to replace matched patterns with the desired format
    def format_time_pattern(self, match: re.Match) -> str:
        """
        Given a regex match of e.g. '0800', returns '08:00'.
        """
        return f"{match.group(1)}:{match.group(2)}"

    # Function to replace "-" with " and "
    def replace_dash(self, match: re.Match) -> str:
        """
        Replaces hyphens in a time range with ' and ', e.g. '9:00 AM-11:00 AM' -> '9:00 AM and 11:00 AM'.
        """
        return match.group(0).replace('-', ' and ')

    def clean_duration_expressions(self, text: str) -> str:
        """
        Removes standalone duration expressions like '1 hour', '30 minutes' from the input text,
        without affecting valid time or date expressions.

        Args:
            text: The input text containing potential duration expressions.

        Returns:
            A cleaned text with standalone duration expressions removed.
        """
        duration_patterns = [
            r'\b\d+\s*hours?\b(?![\s-]*\d)',    # e.g., '1 hour', '2 hours' (but not part of a range like '0800-1200')
            r'\b\d+\s*minutes?\b(?![\s-]*\d)',  # e.g., '30 minutes'
            r'\b\d+\s*seconds?\b(?![\s-]*\d)',  # e.g., '45 seconds'
            r'\b\d+\s*days?\b(?![\s-]*\d)',     # e.g., '3 days'
            r'\b\d+\s*weeks?\b(?![\s-]*\d)',    # e.g., '2 weeks'
            r'\b\d+\s*months?\b(?![\s-]*\d)',   # e.g., '5 months'
            r'\b\d+\s*years?\b(?![\s-]*\d)'     # e.g., '10 years'
        ]

        for pattern in duration_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def remove_random_alphanumeric(self, text: str) -> str:
        """
        Removes standalone alphanumeric words like '4hdf8', excluding legitimate time/date patterns.

        Args:
            text: The input text.

        Returns:
            A cleaned text with extraneous alphanumeric words removed.
        """
        valid_patterns = [
            r'\b\d{1,2}(:\d{2})?[APMapm]\b',       # 1-12AM/PM or HH:MMAM/PM
            r'\b\d{1,2}-\d{1,2}[APMapm]\b',        # 1-12AM/PM
            r'\b\d{1,2}\s*-\s*\d{1,2}[APMapm]\b',  # 1-12 AM/PM (with spaces around dash)
            r'\b\d{1,2}(?:st|nd|rd|th)?\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
               r'January|February|March|April|May|June|July|August|September|October|November|December)\b',
            # 1-31 followed by month
            r'\b\d{1,2}(?:AM|PM)?\s*-\s*\d{1,2}(?:AM|PM)?\b',  # 1AM-12PM or 1-12AM/PM
        ]

        placeholders = []
        # Temporarily replace valid date/time matches with placeholders
        for pattern in valid_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                # If a pattern has multiple capture groups (like the month name), flatten them
                if isinstance(match, tuple):
                    match = ''.join(match)
                placeholder = f"PLACEHOLDER_{len(placeholders)}"
                placeholders.append((placeholder, match))
                text = text.replace(match, placeholder)

        # Define pattern for extraneous alphanumeric words
        non_legit_pattern = r'\b\d+[a-zA-Z]+\b'
        text = re.sub(non_legit_pattern, '', text, flags=re.IGNORECASE)

        # Restore placeholders
        for placeholder, match in placeholders:
            text = text.replace(placeholder, match)

        return text.strip()

    def parse_datetime(self, unstructured_datetime: str):
        """
        Parses a date/time string from text and returns a list of datetime objects.

        Args:
            unstructured_datetime: A string containing a human-readable date/time expression.

        Returns:
            A list of datetime objects representing the parsed date/time.
        """
        # Clean duration expressions first
        unstructured_datetime = self.clean_duration_expressions(unstructured_datetime)
        datetime_match = df.find_dates(unstructured_datetime)
        return list(datetime_match)

    def format_datetime(self, input_datetime: datetime.datetime) -> dict:
        """
        Formats a datetime object into separate date and time dictionaries.

        Args:
            input_datetime: A datetime object.

        Returns:
            A dictionary containing date (MM/DD/YYYY) and time info.
        """
        date_str = input_datetime.strftime("%m/%d/%Y")
        time_dict = {
            "time": input_datetime.strftime("%I:%M"),
            "military_time": input_datetime.strftime("%H:%M"),
            "meridien": input_datetime.strftime("%p"),
            "timezone": input_datetime.strftime("%Z"),
        }
        return {
            "date": date_str,
            "time": time_dict,
        }

    def format_date(self, input_datetime: datetime.datetime) -> str:
        """
        Formats a datetime object into a date string (MM/DD/YYYY).
        """
        return input_datetime.strftime("%m/%d/%Y")

    def format_time(self, input_datetime: datetime.datetime) -> dict:
        """
        Formats a datetime object into a time dictionary.
        """
        return {
            "time": input_datetime.strftime("%I:%M"),
            "military_time": input_datetime.strftime("%H:%M"),
            "meridien": input_datetime.strftime("%p"),
            "timezone": input_datetime.strftime("%Z"),
        }

    def clean_input_date(self, unstructured_datetime: str) -> str:
        """
        Cleans a date string if it's in MM/DD/YYYY format, ensuring consistent slash separators.
        """
        mm_dd_yyyy_pattern = re.compile(r'\b\d{1,2}\s*[/\s]\s*\d{1,2}\s*[/\s]\s*\d{4}\b')
        if mm_dd_yyyy_pattern.search(unstructured_datetime):
            cleaned_datetime = unstructured_datetime.strip()
            # Replace spaces or other separators with slash
            cleaned_datetime = re.sub(r'(\d)\s*[/\s]\s*(\d)', r'\1/\2', cleaned_datetime)
            cleaned_datetime = re.sub(r'(\d{2})\s+(\d{4})', r'\1/\2', cleaned_datetime)
            return cleaned_datetime
        else:
            return unstructured_datetime

    def strip_pm_from_start_time_range(self, unstructured_datetime: str) -> str:
        """
        Strips away 'PM' from the start of a time range if present,
        e.g. "3 PM - 5 PM" -> "3 - 5 PM" to handle tricky parses.
        """
        time_range_pattern = re.compile(
            r'(\d{1,2}(?::\d{2})?(?:\.\d{2})?)\s*PM\s*-\s*(\d{1,2}(?::\d{2})?(?:\.\d{2})?\s*(?:AM|PM))',
            re.IGNORECASE
        )
        cleaned_datetime = time_range_pattern.sub(r'\1 - \2', unstructured_datetime)
        return cleaned_datetime

    def get_single_date(self, unstructured_datetime: str):
        """
        Attempts to parse a single date (MM/DD/YYYY) from the given unstructured text.
        Returns the first valid date that's not the current date.
        """
        unstructured_datetime = str(unstructured_datetime)  # ensure string

        try:
            unstructured_datetime = self.clean_input_date(unstructured_datetime)
            unstructured_datetime = self.strip_pm_from_start_time_range(unstructured_datetime)
            parsed_datetime = self.parse_datetime(unstructured_datetime)

            # If it's a long string or multiple matches, attempt a narrower parse
            if len(parsed_datetime) >= LEN_PARSED_DATE_THRESHOLD:
                date_patterns = [
                    r'\b\d{1,2}/\d{1,2}/\d{4}\b',           # MM/DD/YYYY
                    r'\b\d{1,2}-\d{1,2}-\d{4}\b',           # MM-DD-YYYY
                    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.? \d{1,2}, \d{4}\b',  # Month DD, YYYY
                    r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.? \d{4}\b',   # DD Month YYYY
                ]

                parsed_datetime_txt = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, unstructured_datetime)
                    # Filter out false positives (e.g., words like '24hours')
                    for match in matches:
                        if not re.search(r'\b(min|hour|sec|day|week|month|year)s?\b', match, re.IGNORECASE):
                            parsed_datetime_txt.append(match)

                merged_dates = ' '.join(parsed_datetime_txt)
                if merged_dates:
                    parsed_datetime = self.parse_datetime(merged_dates)

            if not parsed_datetime:
                warnings.warn(f"Warning: No valid date found in the input string: {unstructured_datetime}")
                return None

            current_date_str = datetime.datetime.now().strftime("%m/%d/%Y")

            # Try the first parsed date
            req_date = self.format_date(parsed_datetime[0])
            if req_date != current_date_str:
                if len(parsed_datetime) > 1:
                    warnings.warn("Warning: More than 1 date / time value found")
                return req_date

            # If the first parsed date is today's date, try others
            for dt in parsed_datetime[1:]:
                req_date = self.format_date(dt)
                if req_date != current_date_str:
                    warnings.warn("Warning: More than 1 date / time value found")
                    return req_date

            # If all parsed dates appear to be today's date, attempt fuzzy parsing
            try:
                fuzzy_parsed = date_parser.parse(unstructured_datetime, fuzzy=True)
                fuzzy_parsed_str = self.format_date(fuzzy_parsed)
                if fuzzy_parsed_str != current_date_str:
                    req_date = fuzzy_parsed_str
            except ValueError:
                pass

            if len(parsed_datetime) > 1:
                warnings.warn("Warning: More than 1 date / time value found")

            return req_date
        except Exception:
            warnings.warn(f"Warning: No valid date found in the input string: {unstructured_datetime}")
            return None

    def get_single_time(self, unstructured_datetime: str) -> dict:
        """
        Attempts to parse a single time from the given unstructured text.
        Returns the last valid time found, with a 12-hour and 24-hour format.
        """
        unstructured_datetime = str(unstructured_datetime)

        try:
            # e.g., "0800" -> "08:00"
            pattern1 = r'\b(\d{2})(00)\b'
            # e.g., "9:00 AM-11:00 AM" -> "9:00 AM and 11:00 AM"
            pattern2 = r'\b(\d{1,2}:\d{2}\s*(?:AM|PM)?\s*-\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)\b'

            if re.search(pattern1, unstructured_datetime):
                unstructured_datetime = re.sub(pattern1, self.format_time_pattern, unstructured_datetime)
            if re.search(pattern2, unstructured_datetime):
                unstructured_datetime = re.sub(pattern2, self.replace_dash, unstructured_datetime)

            parsed_datetime = self.parse_datetime(unstructured_datetime)
            if not parsed_datetime:
                warnings.warn(f"Warning: No valid time found in the input string: {unstructured_datetime}")
                return {
                    "time": None,
                    "military_time": None,
                    "meridien": None,
                    "timezone": None,
                }

            # Try the last parsed time first
            req_time_dict = self.format_time(parsed_datetime[-1])
            if req_time_dict["military_time"] != "00:00":
                if len(parsed_datetime) > 1:
                    warnings.warn("Warning: More than 1 date / time value found")
                return req_time_dict

            # Attempt fuzzy parsing if the last time was midnight (00:00)
            try:
                fuzzy_parsed_datetime = date_parser.parse(unstructured_datetime, fuzzy=True)
                fuzzy_time_dict = self.format_time(fuzzy_parsed_datetime)
                if fuzzy_time_dict["military_time"] != "00:00":
                    return fuzzy_time_dict
            except ValueError:
                pass

            # Check other parsed times for a time that is not 00:00
            for dt in reversed(parsed_datetime):
                req_time_dict = self.format_time(dt)
                if req_time_dict["military_time"] != "00:00":
                    warnings.warn("Warning: More than 1 date / time value found")
                    return req_time_dict

            if len(parsed_datetime) > 1:
                warnings.warn("Warning: More than 1 date / time value found")

            # Fallback if everything is 00:00
            return {
                "time": None,
                "military_time": None,
                "meridien": None,
                "timezone": None,
            }
        except Exception:
            warnings.warn(f"Warning: No valid time found in the input string: {unstructured_datetime}")
            return {
                "time": None,
                "military_time": None,
                "meridien": None,
                "timezone": None,
            }

    def time_edgecase_no_colon(self, unstructured_datetime: str) -> str:
        """
        Handles edge cases where time is in the format "CHECK IN@1704" or "CHECK OUT1704" 
        and converts it to a format with a colon, e.g. "CHECK IN 17:04".
        """
        unstructured_datetime = str(unstructured_datetime)
        try:
            pattern = r'CHECK\s*(?:IN|OUT)\s*[@:\s-]*\s*(\d{2})(\d{2})'
            return re.sub(pattern, r'CHECK \g<1>:\g<2>', unstructured_datetime, flags=re.IGNORECASE)
        except Exception:
            warnings.warn(f"Warning: No valid time found in the input string: {unstructured_datetime}")
            return unstructured_datetime

    def time_edgecase_four_digit_no_colon(self, unstructured_datetime: str) -> str:
        """
        Handles edge cases where a 3 or 4-digit chunk might represent a valid time (e.g. '1704' -> '17:04').
        Also handles '145PM' -> '01:45PM'.

        Returns:
            A string with the time corrected to contain a colon if valid.
        """
        unstructured_datetime = str(unstructured_datetime).strip().upper()

        try:
            # Check for 3/4-digit times with optional AM/PM (e.g., "145PM" -> "01:45PM")
            extra_pattern = r'^(\d{1,2})(\d{2})(AM|PM)?$'
            match = re.match(extra_pattern, unstructured_datetime)
            if match:
                hours, minutes, meridiem = match.groups()
                hours = int(hours)
                minutes = int(minutes)
                formatted_time = f"{hours:02}:{minutes:02}{meridiem if meridiem else ''}"
                return formatted_time

            # For within a larger text, look for patterns like '17 04', '1704', etc.
            pattern = r'(\d{2})[^\d]*(\d{2})'
            matches = re.findall(pattern, unstructured_datetime)
            for h, m in matches:
                if 0 <= int(h) <= 23 and 0 <= int(m) <= 59:
                    return f"{h}:{m}"

            return unstructured_datetime
        except Exception:
            warnings.warn(f"Warning: No valid time found in the input string: {unstructured_datetime}")
            return unstructured_datetime

    def format_time_range(self, match: re.Match) -> str:
        """
        Normalizes a matched time range group into a "startTime-endTime" format, 
        adding :00 if missing, etc.
        """
        start, start_period, end, end_period = match.groups()
        start = start.strip().replace('.', ':')
        end = end.strip().replace('.', ':')
        start_period = start_period.strip().upper() if start_period else ''
        end_period = end_period.strip().upper() if end_period else ''

        if ':' not in start:
            start += ':00'
        if ':' not in end:
            end += ':00'

        start_time = f"{start}{start_period}"
        end_time = f"{end}{end_period}"
        return f"{start_time}-{end_time}"

    def time_edgecase_no_colon_with_dash(self, unstructured_datetime: str) -> str:
        """
        Normalizes time ranges (e.g., "9-2PM", "9am-2pm", "9 AM - 2 PM") to a consistent 
        "9:00AM-2:00PM" style format.
        """
        unstructured_datetime = str(unstructured_datetime)
        unstructured_datetime = self.clean_duration_expressions(unstructured_datetime)
        unstructured_datetime = self.remove_random_alphanumeric(unstructured_datetime)
        unstructured_datetime = self.strip_pm_from_start_time_range(unstructured_datetime)

        try:
            time_pattern = re.compile(r'(\d{1,2}(?::\d{2})?)(AM|PM)?\s*-\s*(\d{1,2}(?::\d{2})?)(AM|PM)?', re.IGNORECASE)
            unstructured_datetime = time_pattern.sub(self.format_time_range, unstructured_datetime)
            return unstructured_datetime
        except Exception:
            warnings.warn(f"Warning: No valid time range found in the input string: {unstructured_datetime}")
            return unstructured_datetime

    def extended_get_single_time(self, unstructured_datetime: str, edgecase_func) -> dict:
        """
        Applies a custom edge-case function to the unstructured time, then attempts to parse a single time.
        """
        unstructured_datetime = edgecase_func(unstructured_datetime)
        return self.get_single_time(unstructured_datetime)

    def extended_get_single_date(self, unstructured_datetime: str, edgecase_func) -> str:
        """
        Applies a custom edge-case function to the unstructured text, then attempts to parse a single date.
        """
        unstructured_datetime = edgecase_func(unstructured_datetime)
        return self.get_single_date(unstructured_datetime)

    def get_date_multiple_logic(self, unstructured_datetime: str, edgecase_func) -> str:
        """
        Tries multiple logic paths to extract a date from unstructured text.
        """
        try:
            date_val = self.extended_get_single_date(unstructured_datetime, edgecase_func)
            if date_val:
                return date_val
        except Exception:
            pass

        try:
            date_val = self.get_single_date(unstructured_datetime)
            if date_val:
                return date_val
        except Exception:
            pass

        return None

    def get_time_multiple_logic(self, unstructured_datetime: str, edgecase_func) -> dict:
        """
        Tries multiple logic paths to extract a time from unstructured text.
        """
        try:
            time_val = self.extended_get_single_time(unstructured_datetime, edgecase_func)
            if time_val["time"]:
                return time_val
        except Exception:
            pass

        try:
            time_val = self.get_single_time(unstructured_datetime)
            if time_val["time"]:
                return time_val
        except Exception:
            pass

        return {
            "time": None,
            "military_time": None,
            "meridien": None,
            "timezone": None,
        }


# ---------------------------
# Example usage:
# parser = DateTimeParser()
# date_result = parser.get_single_date("We will meet on 09/15/2024 at 3PM.")
# print(date_result)  # => "09/15/2024"

# time_result = parser.get_single_time("The event starts at 3PM-5PM.")
# print(time_result)  # => {"time": "03:00", "military_time": "15:00", "meridien": "PM", "timezone": ""}
# ---------------------------
