# app/services/alert_manager.py
"""
Alert Manager Service - WebSocket Version
Handles alert detection and email notifications
FIXED: Email sending in background thread without asyncio issues
"""

import logging
import smtplib
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from app.config import settings
from app.schemas.alerts import (
    Alert,
    AlertConfiguration,
)

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert detection, throttling, and notifications

    Features:
    - Rule-based alert detection
    - Alert cooldown/throttling to prevent spam
    - Email notifications
    - Alert history tracking
    - Returns alerts for WebSocket broadcasting
    """

    def __init__(self):
        # Alert cooldown tracking: {camera_id: {alert_key: last_alert_time}}
        self.alert_cooldowns: Dict[str, Dict[str, datetime]] = defaultdict(dict)

        # Alert history for statistics
        self.alert_history: Dict[str, List[Alert]] = defaultdict(list)
        self.max_history_per_camera = 100

    # ==================== ALERT DETECTION ====================

    def check_speed_alerts(
        self,
        camera_id: str,
        camera_name: str,
        alert_config: AlertConfiguration,
        tracked_object: dict,
    ) -> Optional[Alert]:
        """Check if tracked object triggers a speed alert"""
        if not alert_config.speed_alerts:
            return None

        # Get speed data
        speed_kmh = tracked_object.get("speed_kmh")
        if speed_kmh is None:
            return None

        object_class = tracked_object.get("class_name", "unknown")
        track_id = tracked_object.get("track_id", "unknown")

        # Check each speed rule
        for rule in alert_config.speed_alerts:
            if not rule.enabled:
                continue

            # Check if rule applies to this object class
            if rule.object_class != object_class:
                continue

            # Check condition
            triggered = False
            if rule.condition == "over" and speed_kmh > rule.threshold_kmh:
                triggered = True
            elif rule.condition == "under" and speed_kmh < rule.threshold_kmh:
                triggered = True

            if triggered:
                alert = Alert(
                    alert_id=str(uuid.uuid4())[:12],
                    camera_id=camera_id,
                    camera_name=camera_name,
                    timestamp=datetime.now(),
                    alert_type="speed",
                    object_class=object_class,
                    track_id=track_id,
                    threshold_value=rule.threshold_kmh,
                    actual_value=round(speed_kmh, 2),
                    condition=rule.condition,
                    unit="km/h",
                    bbox=tracked_object.get("bbox"),
                    centroid=tracked_object.get("centroid"),
                )

                return alert

        return None

    def check_tracking_alerts(
        self,
        camera_id: str,
        camera_name: str,
        alert_config: AlertConfiguration,
        tracked_object: dict,
    ) -> Optional[Alert]:
        """Check if tracked object triggers a dwell time alert"""
        if not alert_config.tracking_alerts:
            return None

        # Get time in frame
        time_seconds = tracked_object.get("time_in_frame_seconds")
        if time_seconds is None:
            return None

        object_class = tracked_object.get("class_name", "unknown")
        track_id = tracked_object.get("track_id", "unknown")

        # Check each tracking rule
        for rule in alert_config.tracking_alerts:
            if not rule.enabled:
                continue

            if rule.object_class != object_class:
                continue

            # Check condition
            triggered = False
            if rule.condition == "over" and time_seconds > rule.threshold_seconds:
                triggered = True
            elif rule.condition == "under" and time_seconds < rule.threshold_seconds:
                triggered = True

            if triggered:
                alert = Alert(
                    alert_id=str(uuid.uuid4())[:12],
                    camera_id=camera_id,
                    camera_name=camera_name,
                    timestamp=datetime.now(),
                    alert_type="tracking",
                    object_class=object_class,
                    track_id=track_id,
                    threshold_value=rule.threshold_seconds,
                    actual_value=round(time_seconds, 2),
                    condition=rule.condition,
                    unit="seconds",
                    bbox=tracked_object.get("bbox"),
                    centroid=tracked_object.get("centroid"),
                )

                return alert

        return None

    def check_distance_alerts(
        self,
        camera_id: str,
        camera_name: str,
        alert_config: AlertConfiguration,
        tracked_object: dict,
    ) -> Optional[Alert]:
        """Check if tracked object triggers a distance alert"""
        if not alert_config.distance_alerts:
            return None

        # Get distance from camera
        distance_m = tracked_object.get("distance_from_camera_m")
        if distance_m is None:
            return None

        object_class = tracked_object.get("class_name", "unknown")
        track_id = tracked_object.get("track_id", "unknown")

        # Check each distance rule
        for rule in alert_config.distance_alerts:
            if not rule.enabled:
                continue

            if rule.object_class != object_class:
                continue

            # Check condition
            triggered = False
            if rule.condition == "over" and distance_m > rule.threshold_meters:
                triggered = True
            elif rule.condition == "under" and distance_m < rule.threshold_meters:
                triggered = True

            if triggered:
                alert = Alert(
                    alert_id=str(uuid.uuid4())[:12],
                    camera_id=camera_id,
                    camera_name=camera_name,
                    timestamp=datetime.now(),
                    alert_type="distance",
                    object_class=object_class,
                    track_id=track_id,
                    threshold_value=rule.threshold_meters,
                    actual_value=round(distance_m, 2),
                    condition=rule.condition,
                    unit="meters",
                    bbox=tracked_object.get("bbox"),
                    centroid=tracked_object.get("centroid"),
                )

                return alert

        return None

    # ==================== ALERT THROTTLING ====================

    def _get_alert_key(self, alert: Alert) -> str:
        """Generate unique key for alert throttling"""
        return f"{alert.alert_type}_{alert.object_class}_{alert.track_id}"

    def should_send_alert(
        self, camera_id: str, alert: Alert, cooldown_seconds: int
    ) -> bool:
        """
        Check if alert should be sent (cooldown check)

        Returns:
            True if alert should be sent, False if in cooldown
        """
        alert_key = self._get_alert_key(alert)

        # Check if we've sent this alert recently
        if alert_key in self.alert_cooldowns[camera_id]:
            last_alert_time = self.alert_cooldowns[camera_id][alert_key]
            time_since_last = datetime.now() - last_alert_time

            if time_since_last < timedelta(seconds=cooldown_seconds):
                logger.debug(
                    f"Alert throttled: {alert_key} "
                    f"(cooldown: {cooldown_seconds}s, "
                    f"time_since_last: {time_since_last.total_seconds():.1f}s)"
                )
                return False

        # Update last alert time
        self.alert_cooldowns[camera_id][alert_key] = datetime.now()
        return True

    # ==================== ALERT PROCESSING ====================

    def process_tracking_data(
        self,
        camera_id: str,
        camera_name: str,
        camera_email: Optional[str],
        alert_config: AlertConfiguration,
        tracking_data: dict,
    ) -> List[dict]:
        """
        Process tracking data and return triggered alerts

        Returns:
            List of alert dictionaries to be sent via WebSocket
        """
        alerts_to_send = []

        if not tracking_data or not tracking_data.get("tracked_objects"):
            return alerts_to_send

        tracked_objects = tracking_data["tracked_objects"]

        # Check each tracked object
        for track_id, obj_data in tracked_objects.items():
            # Check speed alerts
            speed_alert = self.check_speed_alerts(
                camera_id, camera_name, alert_config, obj_data
            )
            if speed_alert:
                alert_dict = self._handle_alert(
                    camera_id, camera_email, alert_config, speed_alert
                )
                if alert_dict:
                    alerts_to_send.append(alert_dict)

            # Check tracking/dwell time alerts
            tracking_alert = self.check_tracking_alerts(
                camera_id, camera_name, alert_config, obj_data
            )
            if tracking_alert:
                alert_dict = self._handle_alert(
                    camera_id, camera_email, alert_config, tracking_alert
                )
                if alert_dict:
                    alerts_to_send.append(alert_dict)

            # Check distance alerts
            distance_alert = self.check_distance_alerts(
                camera_id, camera_name, alert_config, obj_data
            )
            if distance_alert:
                alert_dict = self._handle_alert(
                    camera_id, camera_email, alert_config, distance_alert
                )
                if alert_dict:
                    alerts_to_send.append(alert_dict)

        return alerts_to_send

    def _handle_alert(
        self,
        camera_id: str,
        camera_email: Optional[str],
        alert_config: AlertConfiguration,
        alert: Alert,
    ) -> Optional[dict]:
        """
        Handle triggered alert

        Returns:
            Alert dict if should be sent, None if throttled
        """
        # Check cooldown
        if not self.should_send_alert(camera_id, alert, alert_config.cooldown_seconds):
            return None

        logger.info(
            f"🚨 ALERT: {alert.alert_type.upper()} - "
            f"{alert.object_class} {alert.condition} {alert.threshold_value} {alert.unit} "
            f"(actual: {alert.actual_value} {alert.unit})"
        )

        # Add to history
        self.alert_history[camera_id].append(alert)
        if len(self.alert_history[camera_id]) > self.max_history_per_camera:
            self.alert_history[camera_id] = self.alert_history[camera_id][
                -self.max_history_per_camera :
            ]

        # Send email notification (non-blocking) - FIXED VERSION
        if alert_config.email_enabled and camera_email:
            try:
                # Convert Alert object to dict for email
                alert_dict = alert.dict()

                # Run email sending in background thread (no asyncio issues)
                email_thread = threading.Thread(
                    target=self.send_email_alert,
                    args=(alert_dict, camera_email),
                    daemon=True,  # Thread will terminate when main program exits
                )
                email_thread.start()
                logger.info(f"📧 Email alert queued for {camera_email}")
            except Exception as e:
                logger.error(f"Failed to queue email alert: {e}", exc_info=True)

        # Return alert dict for WebSocket
        return alert.dict()

    # ==================== EMAIL NOTIFICATIONS ====================

    def send_email_alert(self, alert_data: dict, email_to: str) -> bool:
        """
        Send email alert synchronously

        This method runs in a background thread, so it's safe to block
        """
        server = None
        try:
            # Use settings object (loaded by pydantic-settings from .env)
            smtp_server = settings.SMTP_SERVER
            smtp_port = settings.SMTP_PORT
            smtp_user = settings.SMTP_USERNAME
            smtp_password = settings.SMTP_PASSWORD
            from_email = settings.SMTP_FROM_EMAIL or smtp_user

            if not smtp_user or not smtp_password:
                logger.warning("⚠️ SMTP credentials not configured")
                return False

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 {alert_data['alert_type'].upper()} Alert"
            msg["From"] = from_email
            msg["To"] = email_to

            html = f"""
                <html>
                  <head></head>
                  <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background-color: #f44336; color: white; padding: 20px; text-align: center;">
                      <h1 style="margin: 0;">🚨 {alert_data["alert_type"].upper()} ALERT</h1>
                    </div>

                    <div style="padding: 20px; background-color: #f5f5f5;">
                      <h2 style="color: #333;">Alert Details</h2>

                      <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #ddd;">
                          <td style="padding: 10px; font-weight: bold;">Camera:</td>
                          <td style="padding: 10px;">{alert_data["camera_name"]}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                          <td style="padding: 10px; font-weight: bold;">Object:</td>
                          <td style="padding: 10px;">{alert_data["object_class"]}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                          <td style="padding: 10px; font-weight: bold;">Track ID:</td>
                          <td style="padding: 10px;">{alert_data["track_id"]}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                          <td style="padding: 10px; font-weight: bold;">Condition:</td>
                          <td style="padding: 10px;">{alert_data["condition"]} {alert_data["threshold_value"]} {alert_data["unit"]}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                          <td style="padding: 10px; font-weight: bold;">Actual Value:</td>
                          <td style="padding: 10px; color: #f44336; font-weight: bold;">{alert_data["actual_value"]:.2f} {alert_data["unit"]}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                          <td style="padding: 10px; font-weight: bold;">Time:</td>
                          <td style="padding: 10px;">{alert_data["timestamp"]}</td>
                        </tr>
                      </table>

                      <div style="margin-top: 20px; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107;">
                        <strong>Note:</strong> This is an automated alert from your camera monitoring system.
                      </div>
                    </div>

                    <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
                      <p>Camera Monitoring System | Alert ID: {alert_data["alert_id"]}</p>
                    </div>
                  </body>
                </html>
                """
            msg.attach(MIMEText(html, "html"))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, [email_to], msg.as_string())
            server.quit()

            logger.info(f"✅ Email sent successfully to {email_to}")
            return True

        except Exception as e:
            logger.error(f"❌ Email failed to {email_to}: {e}", exc_info=True)
            if server:
                try:
                    server.quit()
                except:
                    pass
            return False

    # ==================== STATISTICS ====================

    def get_alert_stats(self, camera_id: str) -> dict:
        """Get alert statistics for a camera"""
        alerts = self.alert_history.get(camera_id, [])

        if not alerts:
            return {
                "camera_id": camera_id,
                "total_alerts": 0,
                "alerts_by_type": {},
                "alerts_by_class": {},
                "last_alert_time": None,
            }

        # Count by type
        by_type = defaultdict(int)
        by_class = defaultdict(int)

        for alert in alerts:
            by_type[alert.alert_type] += 1
            by_class[alert.object_class] += 1

        return {
            "camera_id": camera_id,
            "total_alerts": len(alerts),
            "alerts_by_type": dict(by_type),
            "alerts_by_class": dict(by_class),
            "last_alert_time": alerts[-1].timestamp if alerts else None,
        }

    def get_recent_alerts(self, camera_id: str, limit: int = 10) -> List[Alert]:
        """Get recent alerts for a camera"""
        alerts = self.alert_history.get(camera_id, [])
        return alerts[-limit:]


# Global alert manager instance
alert_manager = AlertManager()
