"""Admin Dashboard REST API and HTML interface.

Simple dashboard for administrators to:
- View pending reservation requests
- See reservation details
- Approve or reject reservations with one click
- RESUME interrupted conversations (HITL)
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.database.sql_db import ParkingDatabase
from src.admin.admin_service import AdminService
from src.utils.logging import logger

# Create router
router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Global instances (set by server on startup)
db: Optional[ParkingDatabase] = None
admin_service: Optional[AdminService] = None
hitl_workflow = None  # Will be set if HITL workflow is used


def init_dashboard(database: ParkingDatabase, service: AdminService, workflow=None):
    """Initialize dashboard with database and service instances."""
    global db, admin_service, hitl_workflow
    db = database
    admin_service = service
    hitl_workflow = workflow
    logger.info("Dashboard initialized" + (" with HITL support" if workflow else ""))


# ==================== API ENDPOINTS ====================

class QuickDecisionRequest(BaseModel):
    """Quick decision from dashboard button."""
    admin_name: str = "Dashboard Admin"
    reason: Optional[str] = None
    notes: Optional[str] = None


@router.get("/api/reservations/pending")
async def get_pending_reservations():
    """Get all pending reservations for the dashboard."""
    if not admin_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    pending = admin_service.get_pending_reservations()

    # Enrich with full details
    reservations = []
    for res in pending:
        details = admin_service.get_reservation_details(res["id"])
        if details:
            reservations.append({
                "id": details["id"],
                "user_name": details["user_name"],
                "user_surname": details["user_surname"],
                "car_number": details["car_number"],
                "parking_id": details["parking_id"],
                "parking_name": details.get("parking_name", details["parking_id"]),
                "start_time": details["start_time"].isoformat() if isinstance(details["start_time"], datetime) else str(details["start_time"]),
                "end_time": details["end_time"].isoformat() if isinstance(details["end_time"], datetime) else str(details["end_time"]),
                "created_at": details["created_at"].isoformat() if isinstance(details["created_at"], datetime) else str(details["created_at"]),
                "status": details["status"],
            })

    return {"count": len(reservations), "reservations": reservations}


@router.post("/api/reservations/{reservation_id}/approve")
async def approve_reservation(reservation_id: str, request: QuickDecisionRequest):
    """Quick approve a reservation and NOTIFY the waiting chatbot."""
    if not admin_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    # First, update the database
    result = admin_service.approve_reservation(
        reservation_id,
        request.admin_name,
        request.notes
    )

    if result["success"]:
        logger.info(f"Dashboard: Approved {reservation_id} by {request.admin_name}")

        # If HITL workflow is active, update the thread file so chatbot sees it
        conversation_notified = False
        if hitl_workflow:
            try:
                # Update the thread store file - chatbot will detect this
                hitl_workflow.thread_store.update_with_admin_decision(
                    reservation_id=reservation_id,
                    decision="approve",
                    admin_name=request.admin_name,
                )
                conversation_notified = True
                logger.info(f"HITL: Notified chatbot about approval for {reservation_id}")
            except Exception as e:
                logger.warning(f"Could not notify HITL chatbot: {e}")

        return {
            "success": True,
            "message": result["message"],
            "conversation_notified": conversation_notified,
        }
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@router.post("/api/reservations/{reservation_id}/reject")
async def reject_reservation(reservation_id: str, request: QuickDecisionRequest):
    """Quick reject a reservation and NOTIFY the waiting chatbot."""
    if not admin_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    reason = request.reason or "Rejected via admin dashboard"

    # First, update the database
    result = admin_service.reject_reservation(
        reservation_id,
        request.admin_name,
        reason
    )

    if result["success"]:
        logger.info(f"Dashboard: Rejected {reservation_id} by {request.admin_name}")

        # If HITL workflow is active, update the thread file so chatbot sees it
        conversation_notified = False
        if hitl_workflow:
            try:
                # Update the thread store file - chatbot will detect this
                hitl_workflow.thread_store.update_with_admin_decision(
                    reservation_id=reservation_id,
                    decision="reject",
                    admin_name=request.admin_name,
                    reason=reason,
                )
                conversation_notified = True
                logger.info(f"HITL: Notified chatbot about rejection for {reservation_id}")
            except Exception as e:
                logger.warning(f"Could not notify HITL chatbot: {e}")

        return {
            "success": True,
            "message": result["message"],
            "conversation_notified": conversation_notified,
        }
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@router.get("/api/reservations/history")
async def get_reservation_history():
    """Get recently reviewed reservations."""
    if not admin_service:
        raise HTTPException(status_code=500, detail="Service not initialized")

    history = admin_service.get_reviewed_history(20)

    return {
        "count": len(history),
        "reservations": [
            {
                "id": res["id"],
                "user_name": res.get("user_name", ""),
                "user_surname": res.get("user_surname", ""),
                "status": "APPROVED" if res.get("status") == "confirmed" else "REJECTED",
                "reviewed_at": res.get("reviewed_at").isoformat() if isinstance(res.get("reviewed_at"), datetime) else str(res.get("reviewed_at", "")),
                "admin": res.get("approved_by_admin", "N/A"),
            }
            for res in history
        ]
    }


# ==================== HTML DASHBOARD ====================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parking Admin Dashboard</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        header p {
            color: #888;
            font-size: 1.1rem;
        }

        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .stat-card h3 {
            font-size: 2.5rem;
            color: #4facfe;
        }

        .stat-card p {
            color: #888;
            margin-top: 5px;
        }

        .section {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .section h2 {
            margin-bottom: 20px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section h2 .badge {
            background: #4facfe;
            color: #000;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }

        .reservation-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .reservation-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }

        .reservation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .reservation-id {
            font-family: monospace;
            font-size: 1.1rem;
            color: #4facfe;
        }

        .reservation-time {
            color: #888;
            font-size: 0.9rem;
        }

        .reservation-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .detail-item {
            background: rgba(0,0,0,0.2);
            padding: 12px;
            border-radius: 8px;
        }

        .detail-label {
            color: #888;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .detail-value {
            color: #fff;
            font-size: 1.1rem;
            margin-top: 5px;
        }

        .reservation-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }

        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }

        .btn-approve {
            background: linear-gradient(90deg, #00b09b, #96c93d);
            color: #000;
        }

        .btn-approve:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,176,155,0.4);
        }

        .btn-reject {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
            color: #fff;
        }

        .btn-reject:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(255,65,108,0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        .empty-state {
            text-align: center;
            padding: 50px;
            color: #666;
        }

        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }

        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 10px;
        }

        .status-approved {
            color: #00b09b;
            font-weight: bold;
        }

        .status-rejected {
            color: #ff416c;
            font-weight: bold;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            color: #fff;
            font-weight: 500;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
        }

        .toast.show {
            opacity: 1;
        }

        .toast.success {
            background: linear-gradient(90deg, #00b09b, #96c93d);
        }

        .toast.error {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
        }

        /* Modal for rejection reason */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: #1a1a2e;
            padding: 30px;
            border-radius: 16px;
            max-width: 400px;
            width: 90%;
            border: 1px solid rgba(255,255,255,0.1);
        }

        .modal h3 {
            margin-bottom: 20px;
            color: #ff416c;
        }

        .modal textarea {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
            color: #fff;
            font-size: 1rem;
            margin-bottom: 20px;
            resize: vertical;
            min-height: 100px;
        }

        .modal-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }

        .btn-cancel {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }

        .refresh-btn {
            background: rgba(255,255,255,0.1);
            color: #4facfe;
            padding: 8px 15px;
            border: 1px solid #4facfe;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
        }

        .refresh-btn:hover {
            background: #4facfe;
            color: #000;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Parking Admin Dashboard</h1>
            <p>Manage reservation requests in real-time</p>
        </header>

        <div class="stats">
            <div class="stat-card">
                <h3 id="pending-count">-</h3>
                <p>Pending Requests</p>
            </div>
            <div class="stat-card">
                <h3 id="approved-count">-</h3>
                <p>Approved Today</p>
            </div>
            <div class="stat-card">
                <h3 id="rejected-count">-</h3>
                <p>Rejected Today</p>
            </div>
        </div>

        <div class="section">
            <h2>
                Pending Reservations
                <span class="badge" id="pending-badge">0</span>
                <button class="refresh-btn" onclick="loadPendingReservations()">Refresh</button>
            </h2>
            <div id="pending-list">
                <div class="loading">Loading...</div>
            </div>
        </div>

        <div class="section">
            <h2>Recent Activity</h2>
            <div id="history-list">
                <div class="loading">Loading...</div>
            </div>
        </div>
    </div>

    <!-- Toast notification -->
    <div id="toast" class="toast"></div>

    <!-- Rejection modal -->
    <div id="reject-modal" class="modal">
        <div class="modal-content">
            <h3>Reject Reservation</h3>
            <p style="margin-bottom: 15px; color: #888;">
                Please provide a reason for rejection:
            </p>
            <textarea id="reject-reason" placeholder="Enter rejection reason..."></textarea>
            <div class="modal-actions">
                <button class="btn btn-cancel" onclick="closeRejectModal()">Cancel</button>
                <button class="btn btn-reject" onclick="confirmReject()">Reject</button>
            </div>
        </div>
    </div>

    <script>
        let currentRejectId = null;

        // Load pending reservations
        async function loadPendingReservations() {
            try {
                const response = await fetch('/dashboard/api/reservations/pending');
                const data = await response.json();

                document.getElementById('pending-count').textContent = data.count;
                document.getElementById('pending-badge').textContent = data.count;

                const container = document.getElementById('pending-list');

                if (data.count === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                            <p>No pending reservations</p>
                            <p style="font-size: 0.9rem; margin-top: 5px;">All caught up!</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = data.reservations.map(res => `
                    <div class="reservation-card" id="card-${res.id}">
                        <div class="reservation-header">
                            <span class="reservation-id">${res.id}</span>
                            <span class="reservation-time">Created: ${formatDate(res.created_at)}</span>
                        </div>
                        <div class="reservation-details">
                            <div class="detail-item">
                                <div class="detail-label">Name</div>
                                <div class="detail-value">${res.user_name} ${res.user_surname}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Car Number</div>
                                <div class="detail-value">${res.car_number}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Parking</div>
                                <div class="detail-value">${res.parking_name}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Time</div>
                                <div class="detail-value">${formatDate(res.start_time)} - ${formatDate(res.end_time)}</div>
                            </div>
                        </div>
                        <div class="reservation-actions">
                            <button class="btn btn-approve" onclick="approveReservation('${res.id}')">
                                ✓ Approve
                            </button>
                            <button class="btn btn-reject" onclick="openRejectModal('${res.id}')">
                                ✗ Reject
                            </button>
                        </div>
                    </div>
                `).join('');

            } catch (error) {
                console.error('Error loading reservations:', error);
                document.getElementById('pending-list').innerHTML = `
                    <div class="empty-state">
                        <p style="color: #ff416c;">Error loading reservations</p>
                        <button class="refresh-btn" onclick="loadPendingReservations()" style="margin-top: 10px;">
                            Retry
                        </button>
                    </div>
                `;
            }
        }

        // Load history
        async function loadHistory() {
            try {
                const response = await fetch('/dashboard/api/reservations/history');
                const data = await response.json();

                let approved = 0, rejected = 0;
                data.reservations.forEach(res => {
                    if (res.status === 'APPROVED') approved++;
                    else rejected++;
                });

                document.getElementById('approved-count').textContent = approved;
                document.getElementById('rejected-count').textContent = rejected;

                const container = document.getElementById('history-list');

                if (data.count === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <p>No recent activity</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = data.reservations.slice(0, 10).map(res => `
                    <div class="history-item">
                        <div>
                            <span style="color: #4facfe; font-family: monospace;">${res.id}</span>
                            <span style="margin-left: 15px;">${res.user_name} ${res.user_surname}</span>
                        </div>
                        <div>
                            <span class="${res.status === 'APPROVED' ? 'status-approved' : 'status-rejected'}">
                                ${res.status}
                            </span>
                            <span style="color: #666; margin-left: 15px;">by ${res.admin}</span>
                        </div>
                    </div>
                `).join('');

            } catch (error) {
                console.error('Error loading history:', error);
            }
        }

        // Approve reservation
        async function approveReservation(id) {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = 'Processing...';

            try {
                const response = await fetch(`/dashboard/api/reservations/${id}/approve`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ admin_name: 'Dashboard Admin' })
                });

                if (response.ok) {
                    showToast('Reservation approved!', 'success');
                    document.getElementById(`card-${id}`).style.opacity = '0.5';
                    setTimeout(() => {
                        loadPendingReservations();
                        loadHistory();
                    }, 500);
                } else {
                    const error = await response.json();
                    showToast(error.detail || 'Failed to approve', 'error');
                    btn.disabled = false;
                    btn.textContent = '✓ Approve';
                }
            } catch (error) {
                showToast('Network error', 'error');
                btn.disabled = false;
                btn.textContent = '✓ Approve';
            }
        }

        // Open reject modal
        function openRejectModal(id) {
            currentRejectId = id;
            document.getElementById('reject-modal').classList.add('show');
            document.getElementById('reject-reason').focus();
        }

        // Close reject modal
        function closeRejectModal() {
            currentRejectId = null;
            document.getElementById('reject-modal').classList.remove('show');
            document.getElementById('reject-reason').value = '';
        }

        // Confirm reject
        async function confirmReject() {
            const reason = document.getElementById('reject-reason').value.trim();
            if (!reason) {
                alert('Please provide a rejection reason');
                return;
            }

            try {
                const response = await fetch(`/dashboard/api/reservations/${currentRejectId}/reject`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        admin_name: 'Dashboard Admin',
                        reason: reason
                    })
                });

                if (response.ok) {
                    showToast('Reservation rejected', 'success');
                    closeRejectModal();
                    document.getElementById(`card-${currentRejectId}`).style.opacity = '0.5';
                    setTimeout(() => {
                        loadPendingReservations();
                        loadHistory();
                    }, 500);
                } else {
                    const error = await response.json();
                    showToast(error.detail || 'Failed to reject', 'error');
                }
            } catch (error) {
                showToast('Network error', 'error');
            }
        }

        // Show toast notification
        function showToast(message, type) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast show ${type}`;
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        // Format date
        function formatDate(dateStr) {
            if (!dateStr) return 'N/A';
            const date = new Date(dateStr);
            return date.toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }

        // Auto-refresh every 30 seconds
        setInterval(() => {
            loadPendingReservations();
            loadHistory();
        }, 30000);

        // Initial load
        loadPendingReservations();
        loadHistory();
    </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the admin dashboard HTML page."""
    return DASHBOARD_HTML
