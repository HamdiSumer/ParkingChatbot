"""FastAPI REST API server for admin communication."""
import uuid
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.models import (
    ReservationRequest,
    ReservationResponse,
    AdminDecisionRequest,
    AdminDecisionResponse,
    ReservationDetails,
    PendingReservationsList,
    ReservationStatus,
    NotificationMessage,
)
from src.database.sql_db import ParkingDatabase
from src.admin.admin_service import AdminService
from src.api.notification import NotificationService
from src.api.dashboard import router as dashboard_router, init_dashboard
from src.agents.hitl_workflow import HITLWorkflow
from src.utils.logging import logger


# Global instances (initialized on startup)
db: Optional[ParkingDatabase] = None
admin_service: Optional[AdminService] = None
notification_service: Optional[NotificationService] = None
hitl_workflow: Optional[HITLWorkflow] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global db, admin_service, notification_service, hitl_workflow

    # Startup
    logger.info("Starting Admin API server...")
    db = ParkingDatabase()
    admin_service = AdminService(db)
    notification_service = NotificationService()

    # Create SQL agent for answering data questions
    from src.rag.sql_agent import create_sql_agent
    sql_agent = create_sql_agent(db)

    # Initialize HITL workflow with SQL agent for handling questions
    hitl_workflow = HITLWorkflow(db=db, sql_agent=sql_agent)

    # Initialize dashboard with HITL support
    init_dashboard(db, admin_service, hitl_workflow)

    logger.info("Admin API server started with HITL support")
    logger.info("Dashboard available at: http://localhost:8001/dashboard/")

    yield

    # Shutdown
    logger.info("Shutting down Admin API server...")
    if db:
        db.close()
    logger.info("Admin API server stopped")


# Create FastAPI app
app = FastAPI(
    title="Parking Reservation Admin API",
    description="REST API for admin communication and reservation management",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include dashboard router
app.include_router(dashboard_router)


# ==================== ROOT REDIRECT ====================

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to dashboard."""
    return RedirectResponse(url="/dashboard/")


# ==================== RESERVATION ENDPOINTS ====================

@app.post("/api/reservations", response_model=ReservationResponse)
async def create_reservation(
    request: ReservationRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new reservation request.

    This endpoint is called by the User Agent when a user wants to make a reservation.
    The reservation is created with 'pending' status and admin is notified.
    """
    try:
        res_id = f"RES_{uuid.uuid4().hex[:8].upper()}"

        success = db.create_reservation(
            res_id=res_id,
            user_name=request.user_name,
            user_surname=request.user_surname,
            car_number=request.car_number,
            parking_id=request.parking_id,
            start_time=request.start_time,
            end_time=request.end_time,
        )

        if success:
            # Notify admin in background
            background_tasks.add_task(
                notification_service.notify_new_reservation,
                res_id,
                request.user_name,
                request.user_surname,
                request.parking_id,
            )

            logger.info(f"Reservation {res_id} created via API")
            return ReservationResponse(
                success=True,
                reservation_id=res_id,
                message=f"Reservation {res_id} created and pending admin approval",
                status=ReservationStatus.PENDING,
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create reservation")

    except Exception as e:
        logger.error(f"Error creating reservation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reservations/{reservation_id}", response_model=ReservationDetails)
async def get_reservation(reservation_id: str):
    """Get details of a specific reservation."""
    reservation = admin_service.get_reservation_details(reservation_id)

    if not reservation:
        raise HTTPException(status_code=404, detail=f"Reservation {reservation_id} not found")

    return ReservationDetails(
        id=reservation["id"],
        user_name=reservation["user_name"],
        user_surname=reservation["user_surname"],
        car_number=reservation["car_number"],
        parking_id=reservation["parking_id"],
        parking_name=reservation.get("parking_name"),
        start_time=reservation["start_time"],
        end_time=reservation["end_time"],
        status=ReservationStatus(reservation["status"]),
        created_at=reservation["created_at"],
        reviewed_at=reservation.get("reviewed_at"),
        approved_by_admin=reservation.get("approved_by_admin"),
        rejection_reason=reservation.get("rejection_reason"),
        admin_notes=reservation.get("admin_notes"),
    )


@app.get("/api/reservations/pending/list", response_model=PendingReservationsList)
async def list_pending_reservations():
    """Get all pending reservations awaiting admin review."""
    pending = admin_service.get_pending_reservations()

    reservations = []
    for res in pending:
        details = admin_service.get_reservation_details(res["id"])
        if details:
            reservations.append(ReservationDetails(
                id=details["id"],
                user_name=details["user_name"],
                user_surname=details["user_surname"],
                car_number=details["car_number"],
                parking_id=details["parking_id"],
                parking_name=details.get("parking_name"),
                start_time=details["start_time"],
                end_time=details["end_time"],
                status=ReservationStatus(details["status"]),
                created_at=details["created_at"],
            ))

    return PendingReservationsList(count=len(reservations), reservations=reservations)


# ==================== ADMIN DECISION ENDPOINTS ====================

@app.post("/api/admin/decision", response_model=AdminDecisionResponse)
async def admin_decision(
    request: AdminDecisionRequest,
    background_tasks: BackgroundTasks
):
    """
    Process admin's decision on a reservation.

    This endpoint is called by the Admin Agent or Admin CLI when admin
    approves or rejects a reservation.
    """
    decision = request.decision.lower()

    if decision not in ["approve", "reject"]:
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'approve' or 'reject'"
        )

    if decision == "reject" and not request.reason:
        raise HTTPException(
            status_code=400,
            detail="Reason is required when rejecting a reservation"
        )

    try:
        if decision == "approve":
            result = admin_service.approve_reservation(
                request.reservation_id,
                request.admin_name,
                request.notes
            )
            new_status = ReservationStatus.CONFIRMED if result["success"] else None
        else:
            result = admin_service.reject_reservation(
                request.reservation_id,
                request.admin_name,
                request.reason
            )
            new_status = ReservationStatus.REJECTED if result["success"] else None

        if result["success"]:
            # Notify about decision in background
            background_tasks.add_task(
                notification_service.notify_decision,
                request.reservation_id,
                decision,
                request.admin_name,
                request.reason,
            )

        return AdminDecisionResponse(
            success=result["success"],
            message=result["message"],
            reservation_id=request.reservation_id,
            new_status=new_status,
        )

    except Exception as e:
        logger.error(f"Error processing admin decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NOTIFICATION ENDPOINTS ====================

@app.get("/api/notifications/pending", response_model=List[NotificationMessage])
async def get_pending_notifications():
    """
    Get pending notifications for admin.

    This endpoint is polled by admin clients to check for new reservation requests.
    """
    return notification_service.get_pending_notifications()


@app.delete("/api/notifications/{notification_id}")
async def acknowledge_notification(notification_id: str):
    """Acknowledge and remove a notification."""
    notification_service.acknowledge(notification_id)
    return {"success": True, "message": f"Notification {notification_id} acknowledged"}


# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "parking-admin-api",
    }


# ==================== RUN SERVER ====================

def run_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
