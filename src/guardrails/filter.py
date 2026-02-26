"""Data protection and guard rails for preventing sensitive data exposure."""
import re
from typing import Tuple
from src.utils.logging import logger


class DataProtectionFilter:
    """Filter to protect sensitive data and detect potentially harmful requests."""

    # Patterns for detecting sensitive information
    SENSITIVE_PATTERNS = {
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "phone": r"\b(?:\+1|1)?[-.\s]?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "password": r"(?i)(password|pwd|pass)\s*[:=]\s*\S+",
        "api_key": r"(?i)(api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*\S+",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "sql_injection": r"(?i)\s*('|\")\s*(OR|AND|1|0)\s*('|\")\s*=",  # ' OR '1'='1, ' OR '='  type patterns
    }

    # Suspicious keywords that might indicate malicious intent
    SUSPICIOUS_KEYWORDS = [
        "admin",
        "database",
        "delete",
        "drop",
        "insert",
        "update",
        "sql",
        "hack",
        "bypass",
        "exploit",
        "injection",
        "shell",
        "execute",
        "system",
        "rm -rf",
        "format",
        "wipe",
    ]

    # Blacklisted operations
    BLACKLISTED_OPERATIONS = [
        "access other users data",
        "modify reservations",
        "access admin panel",
        "change prices",
        "change availability",
        "delete user information",
    ]

    def __init__(self):
        """Initialize the data protection filter."""
        logger.info("Data protection filter initialized")

    def check_safety(self, message: str) -> Tuple[bool, str]:
        """Check if message is safe to process.

        Args:
            message: User message to check.

        Returns:
            Tuple of (is_safe, reason). If not safe, reason explains why.
        """
        # Check for sensitive data
        sensitive_found = self._check_sensitive_data(message)
        if sensitive_found:
            return False, f"Message contains sensitive data: {sensitive_found}"

        # Check for suspicious keywords
        suspicious = self._check_suspicious_intent(message)
        if suspicious:
            return False, f"Detected potentially harmful intent: {suspicious}"

        # Check for blacklisted operations
        blacklisted = self._check_blacklisted_operations(message)
        if blacklisted:
            return False, f"Operation not allowed: {blacklisted}"

        return True, ""

    def _check_sensitive_data(self, message: str) -> str:
        """Check for sensitive data patterns.

        Args:
            message: Message to check.

        Returns:
            Type of sensitive data found, or empty string if none.
        """
        for data_type, pattern in self.SENSITIVE_PATTERNS.items():
            if re.search(pattern, message):
                logger.warning(f"Sensitive data detected: {data_type}")
                return data_type
        return ""

    def _check_suspicious_intent(self, message: str) -> str:
        """Check for suspicious keywords indicating harmful intent.

        Args:
            message: Message to check.

        Returns:
            Suspicious keyword found, or empty string if none.
        """
        message_lower = message.lower()

        suspicious_count = 0
        found_keywords = []

        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in message_lower:
                suspicious_count += 1
                found_keywords.append(keyword)

        # Flag if multiple suspicious keywords found
        if suspicious_count >= 2:
            logger.warning(f"Multiple suspicious keywords detected: {found_keywords}")
            return ", ".join(found_keywords)

        return ""

    def _check_blacklisted_operations(self, message: str) -> str:
        """Check for explicitly blacklisted operations.

        Args:
            message: Message to check.

        Returns:
            Blacklisted operation found, or empty string if none.
        """
        message_lower = message.lower()

        for operation in self.BLACKLISTED_OPERATIONS:
            if operation in message_lower:
                logger.warning(f"Blacklisted operation detected: {operation}")
                return operation

        return ""

    def filter_response(self, response: str) -> str:
        """Filter sensitive data from chatbot response.

        Args:
            response: Chatbot response to filter.

        Returns:
            Filtered response with sensitive data removed/masked.
        """
        filtered = response

        # Mask email addresses
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["email"],
            "[EMAIL_MASKED]",
            filtered,
        )

        # Mask phone numbers
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["phone"],
            "[PHONE_MASKED]",
            filtered,
        )

        # Mask IP addresses
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["ipv4"],
            "[IP_MASKED]",
            filtered,
        )

        # Mask credit cards
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["credit_card"],
            "[CC_MASKED]",
            filtered,
        )

        # Mask SSN
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["ssn"],
            "[SSN_MASKED]",
            filtered,
        )

        # Mask API keys and passwords
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["api_key"],
            "[API_KEY_MASKED]",
            filtered,
        )
        filtered = re.sub(
            self.SENSITIVE_PATTERNS["password"],
            "[PASSWORD_MASKED]",
            filtered,
        )

        if filtered != response:
            logger.info("Response filtered for sensitive data")

        return filtered

    def filter_output(self, response: str) -> Tuple[str, bool]:
        """Comprehensive output filtering for LLM responses.

        Combines PII filtering with checks for:
        - Internal system information leakage
        - Potential hallucinated credentials
        - Database schema/query information
        - File path exposure

        Args:
            response: LLM response to filter.

        Returns:
            Tuple of (filtered_response, was_modified).
        """
        original = response
        filtered = self.filter_response(response)

        # Additional output-specific patterns
        output_patterns = {
            # File paths that might reveal system structure
            "file_path": (r"(?:/[a-zA-Z0-9_.-]+){3,}", "[PATH_MASKED]"),
            # SQL table/column references that shouldn't be exposed
            "sql_internal": (r"(?i)\b(SELECT|FROM|WHERE|JOIN)\s+[a-z_]+\s+(FROM|WHERE|ON|=)", "[QUERY_MASKED]"),
            # Environment variable patterns
            "env_var": (r"(?i)(DB_|API_|SECRET_|PASSWORD_|KEY_)[A-Z_]+=\S+", "[ENV_MASKED]"),
            # Connection strings
            "conn_string": (r"(?i)(mysql|postgres|mongodb|redis)://[^\s]+", "[CONN_MASKED]"),
            # Internal error messages with stack traces
            "stack_trace": (r"(?:Traceback|File \"|at \w+\.\w+\()", "[ERROR_MASKED]"),
        }

        for pattern_name, (pattern, replacement) in output_patterns.items():
            if re.search(pattern, filtered):
                filtered = re.sub(pattern, replacement, filtered)
                logger.warning(f"Output contained {pattern_name}, masked")

        # Check for potential jailbreak responses
        jailbreak_indicators = [
            "ignore previous instructions",
            "disregard all prior",
            "bypass security",
            "here is the database schema",
            "admin credentials",
            "root password",
        ]

        response_lower = filtered.lower()
        for indicator in jailbreak_indicators:
            if indicator in response_lower:
                logger.warning(f"Potential jailbreak response detected: {indicator}")
                filtered = "I cannot provide that information. How can I help you with parking-related questions?"
                break

        was_modified = filtered != original
        if was_modified:
            logger.info("Output filtering applied to response")

        return filtered, was_modified


class PIIDetector:
    """Detect Personally Identifiable Information (PII) in text."""

    PII_CATEGORIES = {
        "name": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
        "car_plate": r"\b[A-Z]{2,3}-\d{3,4}[A-Z]{0,2}\b",
        "phone": r"\b(?:\+1|1)?[-.\s]?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    }

    def __init__(self):
        """Initialize PII detector."""
        logger.info("PII detector initialized")

    def detect_pii(self, text: str) -> dict:
        """Detect PII in text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary with detected PII types and matches.
        """
        detected = {}

        for pii_type, pattern in self.PII_CATEGORIES.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches
                logger.info(f"PII detected - {pii_type}: {len(matches)} instances")

        return detected

    def is_safe_to_log(self, text: str) -> bool:
        """Check if text can be safely logged (doesn't contain PII).

        Args:
            text: Text to check.

        Returns:
            True if safe to log, False otherwise.
        """
        detected = self.detect_pii(text)
        return len(detected) == 0
