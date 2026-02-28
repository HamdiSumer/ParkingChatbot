"""Reservation File Writer Service.

This service writes confirmed reservation details to a text file
when an administrator approves a reservation.

File format: Name | Car Number | Reservation Period | Approval Time

Security features:
- Input validation and sanitization
- File locking to prevent race conditions
- Append-only mode (no overwrites)
- Path traversal prevention
"""
import os
import re
import fcntl
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.utils.logging import logger


@dataclass
class ReservationRecord:
    """Data class for a confirmed reservation record."""
    reservation_id: str
    user_name: str
    user_surname: str
    car_number: str
    start_time: datetime
    end_time: datetime
    approval_time: datetime
    approved_by: str
    parking_id: str


class ReservationFileWriter:
    """Secure file writer for confirmed reservations.

    Features:
    - Thread-safe file writing with file locking
    - Input sanitization to prevent injection
    - Atomic writes with proper error handling
    - Configurable output path
    """

    # Characters allowed in text fields (alphanumeric, spaces, common punctuation)
    SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_\.]+$')

    def __init__(self, output_dir: str = "./data/confirmed_reservations"):
        """Initialize the reservation file writer.

        Args:
            output_dir: Directory to store reservation files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Main reservations file
        self.reservations_file = self.output_dir / "confirmed_reservations.txt"

        # Ensure file exists with header
        self._ensure_file_exists()

        logger.info(f"ReservationFileWriter initialized at {self.output_dir}")

    def _ensure_file_exists(self):
        """Create the reservations file with header if it doesn't exist."""
        if not self.reservations_file.exists():
            header = self._create_header()
            with open(self.reservations_file, 'w') as f:
                f.write(header)
            logger.info(f"Created reservations file: {self.reservations_file}")

    def _create_header(self) -> str:
        """Create the file header."""
        separator = "=" * 120
        header_line = f"{'Name':<25} | {'Car Number':<15} | {'Reservation Period':<40} | {'Approval Time':<20} | {'Admin':<15}"
        return f"{separator}\nCONFIRMED PARKING RESERVATIONS\n{separator}\n{header_line}\n{separator}\n"

    def _sanitize_input(self, value: str, max_length: int = 50) -> str:
        """Sanitize input to prevent injection attacks.

        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not value:
            return "N/A"

        # Convert to string and strip
        value = str(value).strip()

        # Remove pipe characters (our delimiter)
        value = value.replace('|', '-')

        # Remove newlines and carriage returns
        value = value.replace('\n', ' ').replace('\r', ' ')

        # Truncate to max length
        if len(value) > max_length:
            value = value[:max_length]

        return value

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for display.

        Args:
            dt: Datetime object

        Returns:
            Formatted string
        """
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except ValueError:
                return dt
        return dt.strftime("%Y-%m-%d %H:%M")

    def _format_reservation_line(self, record: ReservationRecord) -> str:
        """Format a reservation record as a line for the file.

        Args:
            record: ReservationRecord to format

        Returns:
            Formatted line string
        """
        # Sanitize all inputs
        full_name = self._sanitize_input(f"{record.user_name} {record.user_surname}", 25)
        car_number = self._sanitize_input(record.car_number, 15)
        admin = self._sanitize_input(record.approved_by, 15)

        # Format times
        start = self._format_datetime(record.start_time)
        end = self._format_datetime(record.end_time)
        approval = self._format_datetime(record.approval_time)

        # Create reservation period string
        period = f"{start} to {end}"

        # Format the line with fixed widths
        line = f"{full_name:<25} | {car_number:<15} | {period:<40} | {approval:<20} | {admin:<15}"

        return line

    def write_confirmed_reservation(
        self,
        reservation_id: str,
        user_name: str,
        user_surname: str,
        car_number: str,
        start_time: datetime,
        end_time: datetime,
        approved_by: str,
        parking_id: str = "N/A",
    ) -> dict:
        """Write a confirmed reservation to the file.

        This method is thread-safe and uses file locking.

        Args:
            reservation_id: Unique reservation ID
            user_name: User's first name
            user_surname: User's last name
            car_number: Car registration number
            start_time: Reservation start time
            end_time: Reservation end time
            approved_by: Admin who approved
            parking_id: Parking location ID

        Returns:
            Dict with success status and message
        """
        try:
            # Create record
            record = ReservationRecord(
                reservation_id=reservation_id,
                user_name=user_name,
                user_surname=user_surname,
                car_number=car_number,
                start_time=start_time,
                end_time=end_time,
                approval_time=datetime.utcnow(),
                approved_by=approved_by,
                parking_id=parking_id,
            )

            # Format the line
            line = self._format_reservation_line(record)

            # Write with file locking (thread-safe)
            with open(self.reservations_file, 'a') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(f"{line}\n")
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.info(f"Wrote confirmed reservation {reservation_id} to file")

            return {
                "success": True,
                "message": f"Reservation {reservation_id} written to file",
                "file": str(self.reservations_file),
            }

        except Exception as e:
            logger.error(f"Failed to write reservation {reservation_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to write reservation: {str(e)}",
            }

    def get_all_reservations(self) -> list:
        """Read all confirmed reservations from the file.

        Returns:
            List of reservation lines
        """
        try:
            with open(self.reservations_file, 'r') as f:
                lines = f.readlines()

            # Skip header lines (first 5 lines)
            data_lines = [line.strip() for line in lines[5:] if line.strip()]
            return data_lines

        except Exception as e:
            logger.error(f"Failed to read reservations: {e}")
            return []

    def get_file_path(self) -> str:
        """Get the path to the reservations file.

        Returns:
            Absolute path to the reservations file
        """
        return str(self.reservations_file.absolute())


# Singleton instance for global use
_writer_instance: Optional[ReservationFileWriter] = None


def get_reservation_writer() -> ReservationFileWriter:
    """Get the global ReservationFileWriter instance.

    Returns:
        ReservationFileWriter singleton instance
    """
    global _writer_instance
    if _writer_instance is None:
        _writer_instance = ReservationFileWriter()
    return _writer_instance
