"""Notification service for admin alerts."""
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque
import httpx

from src.utils.logging import logger


class NotificationService:
    """Service for managing admin notifications."""

    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize notification service.

        Args:
            webhook_url: Optional webhook URL for external notifications.
        """
        self.webhook_url = webhook_url
        self._pending_notifications: deque = deque(maxlen=100)
        self._notification_callbacks: List[callable] = []
        logger.info("NotificationService initialized")

    def register_callback(self, callback: callable):
        """Register a callback to be called when new notification arrives.

        Args:
            callback: Async function to call with notification data.
        """
        self._notification_callbacks.append(callback)

    async def notify_new_reservation(
        self,
        reservation_id: str,
        user_name: str,
        user_surname: str,
        parking_id: str,
    ):
        """Send notification about new reservation.

        Args:
            reservation_id: The reservation ID.
            user_name: User's first name.
            user_surname: User's last name.
            parking_id: Parking space ID.
        """
        notification = {
            "id": str(uuid.uuid4()),
            "type": "new_reservation",
            "reservation_id": reservation_id,
            "message": f"New reservation request: {reservation_id} from {user_name} {user_surname} for {parking_id}",
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "high",
            "data": {
                "user_name": user_name,
                "user_surname": user_surname,
                "parking_id": parking_id,
            }
        }

        # Add to pending queue
        self._pending_notifications.append(notification)
        logger.info(f"Notification queued: {notification['message']}")

        # Call registered callbacks
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

        # Send to webhook if configured
        if self.webhook_url:
            await self._send_webhook(notification)

        # Print to console for visibility
        self._print_notification(notification)

    async def notify_decision(
        self,
        reservation_id: str,
        decision: str,
        admin_name: str,
        reason: Optional[str] = None,
    ):
        """Send notification about admin decision.

        Args:
            reservation_id: The reservation ID.
            decision: approve or reject.
            admin_name: Name of admin who made decision.
            reason: Rejection reason (if applicable).
        """
        if decision == "approve":
            message = f"Reservation {reservation_id} APPROVED by {admin_name}"
        else:
            message = f"Reservation {reservation_id} REJECTED by {admin_name}: {reason}"

        notification = {
            "id": str(uuid.uuid4()),
            "type": "decision_made",
            "reservation_id": reservation_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": "normal",
            "data": {
                "decision": decision,
                "admin_name": admin_name,
                "reason": reason,
            }
        }

        logger.info(f"Decision notification: {message}")

        # Call registered callbacks
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

        # Send to webhook if configured
        if self.webhook_url:
            await self._send_webhook(notification)

    def get_pending_notifications(self) -> List[Dict]:
        """Get all pending notifications.

        Returns:
            List of pending notification dictionaries.
        """
        return list(self._pending_notifications)

    def acknowledge(self, notification_id: str) -> bool:
        """Acknowledge and remove a notification.

        Args:
            notification_id: ID of notification to acknowledge.

        Returns:
            True if notification was found and removed.
        """
        for i, notif in enumerate(self._pending_notifications):
            if notif.get("id") == notification_id:
                del self._pending_notifications[i]
                logger.info(f"Notification {notification_id} acknowledged")
                return True
        return False

    def clear_all(self):
        """Clear all pending notifications."""
        self._pending_notifications.clear()
        logger.info("All notifications cleared")

    async def _send_webhook(self, notification: Dict):
        """Send notification to webhook URL.

        Args:
            notification: Notification data to send.
        """
        if not self.webhook_url:
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=notification,
                    timeout=10.0,
                )
                if response.status_code == 200:
                    logger.info(f"Webhook notification sent: {notification['id']}")
                else:
                    logger.warning(
                        f"Webhook returned {response.status_code}: {response.text}"
                    )
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def _print_notification(self, notification: Dict):
        """Print notification to console for visibility.

        Args:
            notification: Notification data.
        """
        print("\n" + "=" * 60)
        print("  ADMIN NOTIFICATION")
        print("=" * 60)
        print(f"  Type: {notification['type']}")
        print(f"  Message: {notification['message']}")
        print(f"  Time: {notification['timestamp']}")
        print("=" * 60 + "\n")


# Synchronous wrapper for use in non-async contexts
class SyncNotificationService:
    """Synchronous wrapper for NotificationService."""

    def __init__(self, webhook_url: Optional[str] = None):
        self._async_service = NotificationService(webhook_url)

    def notify_new_reservation(
        self,
        reservation_id: str,
        user_name: str,
        user_surname: str,
        parking_id: str,
    ):
        """Sync wrapper for notify_new_reservation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create task for running loop
            asyncio.create_task(
                self._async_service.notify_new_reservation(
                    reservation_id, user_name, user_surname, parking_id
                )
            )
        else:
            loop.run_until_complete(
                self._async_service.notify_new_reservation(
                    reservation_id, user_name, user_surname, parking_id
                )
            )

    def notify_decision(
        self,
        reservation_id: str,
        decision: str,
        admin_name: str,
        reason: Optional[str] = None,
    ):
        """Sync wrapper for notify_decision."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            asyncio.create_task(
                self._async_service.notify_decision(
                    reservation_id, decision, admin_name, reason
                )
            )
        else:
            loop.run_until_complete(
                self._async_service.notify_decision(
                    reservation_id, decision, admin_name, reason
                )
            )

    def get_pending_notifications(self) -> List[Dict]:
        return self._async_service.get_pending_notifications()

    def acknowledge(self, notification_id: str) -> bool:
        return self._async_service.acknowledge(notification_id)
