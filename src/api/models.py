"""Pydantic models for the REST API."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ReservationStatus(str, Enum):
    """Reservation status enum."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ReservationRequest(BaseModel):
    """Request to create a new reservation."""
    user_name: str = Field(..., description="User's first name")
    user_surname: str = Field(..., description="User's last name")
    car_number: str = Field(..., description="Car registration number")
    parking_id: str = Field(..., description="ID of the parking space")
    start_time: datetime = Field(..., description="Reservation start time")
    end_time: datetime = Field(..., description="Reservation end time")


class ReservationResponse(BaseModel):
    """Response after creating a reservation."""
    success: bool
    reservation_id: Optional[str] = None
    message: str
    status: Optional[ReservationStatus] = None


class AdminDecisionRequest(BaseModel):
    """Admin's decision on a reservation."""
    reservation_id: str = Field(..., description="Reservation ID to act on")
    decision: str = Field(..., description="approve or reject")
    admin_name: str = Field(..., description="Name of the admin making decision")
    reason: Optional[str] = Field(None, description="Reason for rejection (required if rejecting)")
    notes: Optional[str] = Field(None, description="Optional admin notes")


class AdminDecisionResponse(BaseModel):
    """Response after admin decision."""
    success: bool
    message: str
    reservation_id: str
    new_status: Optional[ReservationStatus] = None


class ReservationDetails(BaseModel):
    """Full reservation details."""
    id: str
    user_name: str
    user_surname: str
    car_number: str
    parking_id: str
    parking_name: Optional[str] = None
    start_time: datetime
    end_time: datetime
    status: ReservationStatus
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    approved_by_admin: Optional[str] = None
    rejection_reason: Optional[str] = None
    admin_notes: Optional[str] = None


class PendingReservationsList(BaseModel):
    """List of pending reservations."""
    count: int
    reservations: List[ReservationDetails]


class NotificationMessage(BaseModel):
    """Notification message for admin."""
    type: str = Field(..., description="Type of notification")
    reservation_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: str = Field(default="normal", description="Priority: low, normal, high")


class WebhookConfig(BaseModel):
    """Webhook configuration for notifications."""
    url: str = Field(..., description="Webhook URL to call")
    secret: Optional[str] = Field(None, description="Optional secret for authentication")
    enabled: bool = Field(default=True)
