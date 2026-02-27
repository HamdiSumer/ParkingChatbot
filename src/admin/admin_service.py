"""Admin service layer for reservation management."""
from typing import Optional
from src.database.sql_db import ParkingDatabase
from src.utils.logging import logger


class AdminService:
    """Service layer for admin operations on reservations."""

    def __init__(self, db: ParkingDatabase = None):
        """Initialize admin service.

        Args:
            db: ParkingDatabase instance. Creates new one if not provided.
        """
        self.db = db or ParkingDatabase()
        logger.info("AdminService initialized")

    def get_pending_reservations(self) -> list:
        """Get all pending reservations awaiting admin review.

        Returns:
            List of pending reservation dictionaries.
        """
        reservations = self.db.get_pending_reservations()
        logger.info(f"Retrieved {len(reservations)} pending reservations")
        return reservations

    def approve_reservation(
        self, res_id: str, admin_name: str, notes: str = None
    ) -> dict:
        """Approve a reservation.

        Args:
            res_id: Reservation ID to approve.
            admin_name: Name of admin performing approval.
            notes: Optional notes about the approval.

        Returns:
            Dict with success status and message.
        """
        reservation = self.db.get_reservation(res_id)
        if not reservation:
            logger.warning(f"Approval failed: Reservation {res_id} not found")
            return {"success": False, "message": f"Reservation {res_id} not found"}

        if reservation["status"] != "pending":
            logger.warning(
                f"Approval failed: Reservation {res_id} is not pending (status: {reservation['status']})"
            )
            return {
                "success": False,
                "message": f"Reservation is not pending (current status: {reservation['status']})",
            }

        success = self.db.approve_reservation(res_id, admin_name, notes)
        if success:
            logger.info(f"Reservation {res_id} approved by {admin_name}")
            return {
                "success": True,
                "message": f"Reservation {res_id} approved successfully",
            }
        else:
            logger.error(f"Failed to approve reservation {res_id}")
            return {"success": False, "message": "Failed to approve reservation"}

    def reject_reservation(
        self, res_id: str, admin_name: str, reason: str
    ) -> dict:
        """Reject a reservation.

        Args:
            res_id: Reservation ID to reject.
            admin_name: Name of admin performing rejection.
            reason: Reason for rejection.

        Returns:
            Dict with success status and message.
        """
        if not reason or not reason.strip():
            return {"success": False, "message": "Rejection reason is required"}

        reservation = self.db.get_reservation(res_id)
        if not reservation:
            logger.warning(f"Rejection failed: Reservation {res_id} not found")
            return {"success": False, "message": f"Reservation {res_id} not found"}

        if reservation["status"] != "pending":
            logger.warning(
                f"Rejection failed: Reservation {res_id} is not pending (status: {reservation['status']})"
            )
            return {
                "success": False,
                "message": f"Reservation is not pending (current status: {reservation['status']})",
            }

        success = self.db.reject_reservation(res_id, admin_name, reason)
        if success:
            logger.info(f"Reservation {res_id} rejected by {admin_name}")
            return {
                "success": True,
                "message": f"Reservation {res_id} rejected",
            }
        else:
            logger.error(f"Failed to reject reservation {res_id}")
            return {"success": False, "message": "Failed to reject reservation"}

    def get_reservation_details(self, res_id: str) -> Optional[dict]:
        """Get full details of a reservation.

        Args:
            res_id: Reservation ID.

        Returns:
            Reservation details dict or None if not found.
        """
        reservation = self.db.get_reservation(res_id)
        if reservation:
            parking = self.db.get_parking_space(reservation["parking_id"])
            if parking:
                reservation["parking_name"] = parking["name"]
                reservation["parking_location"] = parking["location"]
        return reservation

    def get_reservation_status(self, res_id: str) -> Optional[dict]:
        """Get status information for a reservation.

        Args:
            res_id: Reservation ID.

        Returns:
            Dict with status info or None if not found.
        """
        reservation = self.db.get_reservation(res_id)
        if not reservation:
            return None

        parking = self.db.get_parking_space(reservation["parking_id"])
        parking_name = parking["name"] if parking else reservation["parking_id"]

        status_info = {
            "id": reservation["id"],
            "status": reservation["status"],
            "parking_name": parking_name,
            "start_time": reservation["start_time"],
            "end_time": reservation["end_time"],
            "created_at": reservation["created_at"],
            "user_name": reservation.get("user_name", ""),
            "user_surname": reservation.get("user_surname", ""),
            "car_number": reservation.get("car_number", ""),
        }

        if reservation["status"] == "confirmed":
            status_info["approved_by"] = reservation["approved_by_admin"]
            status_info["reviewed_at"] = reservation["reviewed_at"]
            if reservation["admin_notes"]:
                status_info["admin_notes"] = reservation["admin_notes"]

        elif reservation["status"] == "rejected":
            status_info["rejected_by"] = reservation["approved_by_admin"]
            status_info["rejection_reason"] = reservation["rejection_reason"]
            status_info["reviewed_at"] = reservation["reviewed_at"]

        return status_info

    def get_reservations_by_user(self, user_name: str, user_surname: str) -> list:
        """Get all reservations for a user.

        Args:
            user_name: User's first name.
            user_surname: User's last name.

        Returns:
            List of reservation dictionaries.
        """
        return self.db.get_reservations_by_user(user_name, user_surname)

    def get_reviewed_history(self, limit: int = 20) -> list:
        """Get recently reviewed reservations.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of reviewed reservation dictionaries.
        """
        return self.db.get_reviewed_reservations(limit)

    def close(self):
        """Close database connection."""
        self.db.close()
