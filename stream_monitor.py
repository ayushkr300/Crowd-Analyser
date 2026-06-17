"""
stream_monitor.py
-----------------
Real-time crowd density monitoring from streaming sources.
Supports HTTP/HTTPS streams and RTSP CCTV feeds with alert capabilities.
"""

import os
import time
import threading
import tempfile
from datetime import datetime
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path

from crowd_analyzer import get_analyzer


@dataclass
class AlertConfig:
    """Alert configuration"""
    enable_alerts: bool = True
    alert_on_risk: str = "HIGH"  # Alert threshold: LOW, MODERATE, HIGH, CRITICAL
    alert_cooldown_sec: int = 60  # Minimum seconds between alerts
    sound_alert: bool = True
    log_alerts: bool = True


@dataclass
class StreamStats:
    """Real-time stream statistics"""
    frames_processed: int = 0
    current_crowd_count: int = 0
    current_density: float = 0.0
    current_risk: str = "LOW"
    peak_count: int = 0
    peak_density: float = 0.0
    highest_risk: str = "LOW"
    last_alert_time: Optional[float] = None
    stream_fps: float = 0.0
    inference_fps: float = 0.0


class StreamMonitor:
    """
    Real-time crowd monitoring from streaming sources.
    
    Supports:
    - HTTP/HTTPS streams (m3u8, mp4, mjpeg)
    - RTSP CCTV feeds
    - Alert system with customizable thresholds
    
    Example
    -------
    monitor = StreamMonitor(stream_url="rtsp://...", area=300.0)
    monitor.start_monitoring()
    # Monitor in background, check alerts via monitor.stats
    """
    
    RISK_HIERARCHY = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    
    def __init__(
        self,
        stream_url: str,
        area: float,
        sample_rate: float = 1.0,
        alert_config: Optional[AlertConfig] = None,
        callback_on_alert: Optional[Callable] = None,
    ):
        """
        Initialize stream monitor.
        
        Parameters
        ----------
        stream_url : str
            URL to stream (RTSP, HTTP, HTTPS)
        area : float
            Real-world area visible in stream (m²)
        sample_rate : float
            Process 1 frame every N seconds (default 1.0)
        alert_config : AlertConfig
            Alert configuration
        callback_on_alert : callable
            Function to call when alert triggered: callback(stats, risk)
        """
        self.stream_url = stream_url
        self.area = area
        self.sample_rate = sample_rate
        self.alert_config = alert_config or AlertConfig()
        self.callback_on_alert = callback_on_alert
        
        self.analyzer = get_analyzer()
        self.stats = StreamStats()
        
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Alert logging
        self.alert_log_path = Path("outputs") / "alerts.log"
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_monitoring(self) -> bool:
        """
        Start monitoring stream in background thread.
        
        Returns
        -------
        bool
            True if stream connected, False otherwise
        """
        if self.is_running:
            return False
        
        # Test connection
        if not self._test_stream_connection():
            return False
        
        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="StreamMonitorThread"
        )
        self.monitor_thread.start()
        return True
    
    def stop_monitoring(self) -> None:
        """Stop monitoring and cleanup resources."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.cap:
            self.cap.release()
    
    def get_stats(self) -> StreamStats:
        """Get current monitoring statistics."""
        return self.stats
    
    # ----------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------
    
    def _test_stream_connection(self) -> bool:
        """Test if stream is accessible."""
        try:
            cap = cv2.VideoCapture(self.stream_url)
            # Try to read first frame
            ret, _ = cap.read()
            cap.release()
            return ret
        except Exception as e:
            print(f"[StreamMonitor] Connection failed: {e}")
            return False
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (runs in background thread)."""
        self.cap = cv2.VideoCapture(self.stream_url)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Fallback
        
        frame_skip = max(1, int(fps * self.sample_rate))
        frame_idx = 0
        last_process_time = time.time()
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    # Stream ended or error, reconnect
                    print("[StreamMonitor] Stream ended, attempting reconnect...")
                    time.sleep(2)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.stream_url)
                    continue
                
                # Update stream FPS
                self.stats.stream_fps = fps
                
                # Process at sample rate
                if frame_idx % frame_skip == 0:
                    self._process_frame(frame)
                
                frame_idx += 1
                
                # Update inference FPS every 30 frames
                if frame_idx % 30 == 0:
                    now = time.time()
                    elapsed = now - last_process_time
                    if elapsed > 0:
                        self.stats.inference_fps = 30 / elapsed
                    last_process_time = now
        
        except Exception as e:
            print(f"[StreamMonitor] Error in monitoring loop: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process single frame for crowd detection."""
        try:
            # Save to temp file for analyzer
            fd_img, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd_img)
            
            cv2.imwrite(tmp_path, frame)
            
            # Run detection
            results = self.analyzer.predict(tmp_path, self.area)
            
            # Update stats
            self.stats.frames_processed += 1
            self.stats.current_crowd_count = results["count"]
            self.stats.current_density = results["density"]
            self.stats.current_risk = results["risk"]
            
            # Update peaks
            if results["count"] > self.stats.peak_count:
                self.stats.peak_count = results["count"]
            if results["density"] > self.stats.peak_density:
                self.stats.peak_density = results["density"]
            if self._is_risk_higher(results["risk"], self.stats.highest_risk):
                self.stats.highest_risk = results["risk"]
            
            # Check alert condition
            self._check_alert(results["risk"])
            
            # Cleanup
            os.unlink(tmp_path)
        
        except Exception as e:
            print(f"[StreamMonitor] Frame processing error: {e}")
    
    def _check_alert(self, current_risk: str) -> None:
        """Check if alert should be triggered."""
        if not self.alert_config.enable_alerts:
            return
        
        # Check if current risk is at or above alert threshold
        if self.RISK_HIERARCHY[current_risk] >= self.RISK_HIERARCHY[self.alert_config.alert_on_risk]:
            # Check cooldown
            now = time.time()
            if self.stats.last_alert_time is None or (now - self.stats.last_alert_time) >= self.alert_config.alert_cooldown_sec:
                self._trigger_alert(current_risk)
                self.stats.last_alert_time = now
    
    def _trigger_alert(self, risk: str) -> None:
        """Trigger alert for high risk condition."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"🚨 CROWD ALERT [{timestamp}]\n"
            f"Risk Level: {risk}\n"
            f"Crowd Count: {self.stats.current_crowd_count}\n"
            f"Density: {self.stats.current_density:.2f} persons/m²"
        )
        
        print(f"\n{message}\n")
        
        # Log alert
        if self.alert_config.log_alerts:
            self._log_alert(message)
        
        # Sound alert
        if self.alert_config.sound_alert:
            self._trigger_sound_alert()
        
        # Custom callback
        if self.callback_on_alert:
            try:
                self.callback_on_alert(self.stats, risk)
            except Exception as e:
                print(f"[StreamMonitor] Callback error: {e}")
    
    def _log_alert(self, message: str) -> None:
        """Log alert to file."""
        try:
            with open(self.alert_log_path, "a") as f:
                f.write(message + "\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"[StreamMonitor] Failed to log alert: {e}")
    
    def _trigger_sound_alert(self) -> None:
        """Trigger sound alert (platform-dependent)."""
        import sys
        try:
            if sys.platform == "darwin":  # macOS
                os.system("afplay /System/Library/Sounds/Alarm.aiff")
            elif sys.platform == "win32":  # Windows
                import winsound
                winsound.Beep(1000, 500)  # 1kHz, 500ms
            else:  # Linux
                os.system("speaker-test -t sine -f 1000 -l 1")
        except Exception as e:
            print(f"[StreamMonitor] Could not play sound: {e}")
    
    @staticmethod
    def _is_risk_higher(risk1: str, risk2: str) -> bool:
        """Check if risk1 is higher than risk2."""
        hierarchy = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
        return hierarchy.get(risk1, 0) > hierarchy.get(risk2, 0)


if __name__ == "__main__":
    # Example usage
    config = AlertConfig(
        enable_alerts=True,
        alert_on_risk="HIGH",
        alert_cooldown_sec=30,
        sound_alert=True,
        log_alerts=True,
    )
    
    monitor = StreamMonitor(
        stream_url="rtsp://example.com/stream",
        area=200.0,
        sample_rate=2.0,
        alert_config=config,
    )
    
    if monitor.start_monitoring():
        print("Monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                stats = monitor.get_stats()
                print(f"Frames: {stats.frames_processed} | "
                      f"Count: {stats.current_crowd_count} | "
                      f"Density: {stats.current_density:.2f} | "
                      f"Risk: {stats.current_risk}")
                time.sleep(5)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("Monitoring stopped.")
    else:
        print("Failed to connect to stream.")
