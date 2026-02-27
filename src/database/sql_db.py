"""SQLite database for dynamic parking data (availability, prices, working hours)."""
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from src.config import get_config
from src.utils.logging import logger

config = get_config()
Base = declarative_base()


class ParkingSpace(Base):
    """Model for parking space information (dynamic data)."""

    __tablename__ = "parking_spaces"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    capacity = Column(Integer, nullable=False)
    available_spaces = Column(Integer, nullable=False)
    price_per_hour = Column(Float, nullable=False)
    is_open = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow)


class WorkingHours(Base):
    """Model for parking working hours."""

    __tablename__ = "working_hours"

    id = Column(String, primary_key=True)
    parking_id = Column(String, nullable=False)
    day_of_week = Column(Integer, nullable=False)  # 0-6 (Monday-Sunday)
    open_time = Column(String, nullable=False)  # HH:MM format
    close_time = Column(String, nullable=False)


class Reservation(Base):
    """Model for parking reservations."""

    __tablename__ = "reservations"

    id = Column(String, primary_key=True)
    user_name = Column(String, nullable=False)
    user_surname = Column(String, nullable=False)
    car_number = Column(String, nullable=False)
    parking_id = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String, default="pending")  # pending, confirmed, completed, cancelled, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    approved_by_admin = Column(String, nullable=True)  # Admin name who approved/rejected
    rejection_reason = Column(String, nullable=True)  # Reason for rejection
    reviewed_at = Column(DateTime, nullable=True)  # When admin reviewed
    admin_notes = Column(String, nullable=True)  # Optional admin notes


class ParkingDatabase:
    """SQLite database manager for dynamic parking data."""

    def __init__(self, db_path: str = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path or config.SQLITE_DB_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)
        self._init_db()
        logger.info(f"Database initialized at {self.db_path}")

    def _init_db(self):
        """Create database tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_session(self):
        """Get a database session.

        Returns:
            SQLAlchemy session.
        """
        return self.Session()

    def add_parking_space(
        self, id: str, name: str, location: str, capacity: int, price_per_hour: float
    ) -> bool:
        """Add a new parking space.

        Args:
            id: Unique parking space identifier.
            name: Name of parking space.
            location: Location/address.
            capacity: Total parking capacity.
            price_per_hour: Price per hour.

        Returns:
            True if successful.
        """
        session = self.get_session()
        try:
            parking = ParkingSpace(
                id=id,
                name=name,
                location=location,
                capacity=capacity,
                available_spaces=capacity,
                price_per_hour=price_per_hour,
            )
            session.add(parking)
            session.commit()
            logger.info(f"Added parking space: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add parking space: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_parking_space(self, parking_id: str) -> dict:
        """Get parking space details.

        Args:
            parking_id: ID of parking space.

        Returns:
            Dictionary with parking space info or None.
        """
        session = self.get_session()
        try:
            parking = session.query(ParkingSpace).filter_by(id=parking_id).first()
            if parking:
                return {
                    "id": parking.id,
                    "name": parking.name,
                    "location": parking.location,
                    "capacity": parking.capacity,
                    "available_spaces": parking.available_spaces,
                    "price_per_hour": parking.price_per_hour,
                    "is_open": parking.is_open,
                }
            return None
        finally:
            session.close()

    def update_availability(self, parking_id: str, available_spaces: int) -> bool:
        """Update available parking spaces.

        Args:
            parking_id: ID of parking space.
            available_spaces: Number of available spaces.

        Returns:
            True if successful.
        """
        session = self.get_session()
        try:
            parking = session.query(ParkingSpace).filter_by(id=parking_id).first()
            if parking:
                parking.available_spaces = available_spaces
                parking.last_updated = datetime.utcnow()
                session.commit()
                logger.info(f"Updated availability for {parking_id}: {available_spaces}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update availability: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def create_reservation(
        self,
        res_id: str,
        user_name: str,
        user_surname: str,
        car_number: str,
        parking_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> bool:
        """Create a new reservation.

        Args:
            res_id: Unique reservation ID.
            user_name: User's first name.
            user_surname: User's last name.
            car_number: Car registration number.
            parking_id: ID of parking space.
            start_time: Reservation start time.
            end_time: Reservation end time.

        Returns:
            True if successful.
        """
        session = self.get_session()
        try:
            reservation = Reservation(
                id=res_id,
                user_name=user_name,
                user_surname=user_surname,
                car_number=car_number,
                parking_id=parking_id,
                start_time=start_time,
                end_time=end_time,
                status="pending",
            )
            session.add(reservation)
            session.commit()
            logger.info(f"Created reservation: {res_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create reservation: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_reservation(self, res_id: str) -> dict:
        """Get reservation details.

        Args:
            res_id: Reservation ID.

        Returns:
            Dictionary with reservation info or None.
        """
        session = self.get_session()
        try:
            res = session.query(Reservation).filter_by(id=res_id).first()
            if res:
                return {
                    "id": res.id,
                    "user_name": res.user_name,
                    "user_surname": res.user_surname,
                    "car_number": res.car_number,
                    "parking_id": res.parking_id,
                    "start_time": res.start_time,
                    "end_time": res.end_time,
                    "status": res.status,
                    "created_at": res.created_at,
                    "approved_by_admin": res.approved_by_admin,
                    "rejection_reason": res.rejection_reason,
                    "reviewed_at": res.reviewed_at,
                    "admin_notes": res.admin_notes,
                }
            return None
        finally:
            session.close()

    def approve_reservation(self, res_id: str, admin_name: str, notes: str = None) -> bool:
        """Approve a reservation (admin action).

        Args:
            res_id: Reservation ID.
            admin_name: Name of admin approving.
            notes: Optional admin notes.

        Returns:
            True if successful.
        """
        session = self.get_session()
        try:
            res = session.query(Reservation).filter_by(id=res_id).first()
            if res:
                res.status = "confirmed"
                res.approved_by_admin = admin_name
                res.reviewed_at = datetime.utcnow()
                if notes:
                    res.admin_notes = notes
                session.commit()
                logger.info(f"Approved reservation: {res_id} by {admin_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to approve reservation: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def reject_reservation(self, res_id: str, admin_name: str, reason: str) -> bool:
        """Reject a reservation (admin action).

        Args:
            res_id: Reservation ID.
            admin_name: Name of admin rejecting.
            reason: Reason for rejection.

        Returns:
            True if successful.
        """
        session = self.get_session()
        try:
            res = session.query(Reservation).filter_by(id=res_id).first()
            if res:
                res.status = "rejected"
                res.approved_by_admin = admin_name
                res.rejection_reason = reason
                res.reviewed_at = datetime.utcnow()
                session.commit()
                logger.info(f"Rejected reservation: {res_id} by {admin_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reject reservation: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_pending_reservations(self) -> list:
        """Get all pending reservations for admin review.

        Returns:
            List of pending reservation dictionaries.
        """
        session = self.get_session()
        try:
            reservations = session.query(Reservation).filter_by(status="pending").all()
            return [
                {
                    "id": res.id,
                    "user_name": res.user_name,
                    "user_surname": res.user_surname,
                    "car_number": res.car_number,
                    "parking_id": res.parking_id,
                    "start_time": res.start_time,
                    "end_time": res.end_time,
                    "created_at": res.created_at,
                }
                for res in reservations
            ]
        finally:
            session.close()

    def get_reservations_by_user(self, user_name: str, user_surname: str) -> list:
        """Get all reservations for a specific user.

        Args:
            user_name: User's first name.
            user_surname: User's last name.

        Returns:
            List of reservation dictionaries.
        """
        session = self.get_session()
        try:
            reservations = (
                session.query(Reservation)
                .filter_by(user_name=user_name, user_surname=user_surname)
                .order_by(Reservation.created_at.desc())
                .all()
            )
            return [
                {
                    "id": res.id,
                    "parking_id": res.parking_id,
                    "start_time": res.start_time,
                    "end_time": res.end_time,
                    "status": res.status,
                    "created_at": res.created_at,
                    "rejection_reason": res.rejection_reason,
                }
                for res in reservations
            ]
        finally:
            session.close()

    def get_reviewed_reservations(self, limit: int = 20) -> list:
        """Get recently reviewed (approved/rejected) reservations.

        Args:
            limit: Maximum number of reservations to return.

        Returns:
            List of reviewed reservation dictionaries.
        """
        session = self.get_session()
        try:
            reservations = (
                session.query(Reservation)
                .filter(Reservation.status.in_(["confirmed", "rejected"]))
                .filter(Reservation.reviewed_at.isnot(None))
                .order_by(Reservation.reviewed_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": res.id,
                    "user_name": res.user_name,
                    "user_surname": res.user_surname,
                    "parking_id": res.parking_id,
                    "status": res.status,
                    "reviewed_at": res.reviewed_at,
                    "approved_by_admin": res.approved_by_admin,
                    "rejection_reason": res.rejection_reason,
                }
                for res in reservations
            ]
        finally:
            session.close()

    def close(self):
        """Close database connection and dispose engine."""
        try:
            self.engine.dispose()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
