"""
test_stream_generator.py
------------------------
Generate a local test stream with simulated crowd data.
Use this to test the Live Stream Monitor without needing external URLs.
"""

import cv2
import numpy as np
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import time
import queue


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving MJPEG stream"""
    
    frame_queue = None
    
    def do_GET(self):
        """Serve MJPEG stream"""
        if self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while True:
                try:
                    # Get frame from queue
                    frame = self.frame_queue.get(timeout=1)
                    
                    # Encode frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_data = buffer.tobytes()
                    
                    # Send frame in MJPEG format
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n')
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Stream error: {e}")
                    break
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def generate_test_stream(output_queue, num_frames=1000, fps=10):
    """
    Generate synthetic test frames with simulated crowd.
    
    Parameters
    ----------
    output_queue : queue.Queue
        Queue to put generated frames into
    num_frames : int
        Number of frames to generate
    fps : int
        Frames per second
    """
    width, height = 640, 480
    frame_delay = 1.0 / fps
    
    print(f"[Stream Generator] Starting: {num_frames} frames @ {fps} FPS")
    
    for frame_num in range(num_frames):
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add text header
        cv2.rectangle(frame, (0, 0), (width, 50), (25, 25, 112), -1)
        cv2.putText(frame, "Test Crowd Stream", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Simulate crowd density that changes over time
        # Pattern: LOW -> MODERATE -> HIGH -> CRITICAL -> repeat
        cycle_position = (frame_num % 400) / 400.0
        
        if cycle_position < 0.25:
            # LOW density (< 2 persons/m²)
            crowd_level = "LOW"
            num_people = 5
            color = (0, 255, 0)  # Green
        elif cycle_position < 0.50:
            # MODERATE density (2-4 persons/m²)
            crowd_level = "MODERATE"
            num_people = 15
            color = (0, 165, 255)  # Orange
        elif cycle_position < 0.75:
            # HIGH density (4-6 persons/m²)
            crowd_level = "HIGH"
            num_people = 30
            color = (0, 100, 255)  # Orange-Red
        else:
            # CRITICAL density (> 6 persons/m²)
            crowd_level = "CRITICAL"
            num_people = 50
            color = (0, 0, 255)  # Red
        
        # Draw simulated crowd (circles representing people)
        np.random.seed(frame_num // 10)  # Consistent positions per ~1 second
        for i in range(num_people):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(70, height - 50)
            radius = np.random.randint(8, 15)
            cv2.circle(frame, (x, y), radius, color, -1)
            # Draw head circles with slight variation
            cv2.circle(frame, (x, y - radius // 2), radius // 2, (100, 100, 150), 2)
        
        # Add info panel
        cv2.rectangle(frame, (0, height - 120), (width, height), (30, 30, 30), -1)
        
        # Status
        status_color = color
        cv2.circle(frame, (width - 30, height - 100), 8, status_color, -1)
        cv2.putText(frame, f"Status: {crowd_level}", (10, height - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Crowd count
        cv2.putText(frame, f"Detected: {num_people} people", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Estimated density (assuming 300 m² area)
        density = (num_people / 300.0)
        cv2.putText(frame, f"Density: {density:.2f} persons/m²", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Frame info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Frame: {frame_num} | {timestamp}", (width - 350, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Put frame in queue
        try:
            output_queue.put(frame, timeout=1)
        except queue.Full:
            # If queue is full, remove oldest frame
            try:
                output_queue.get_nowait()
                output_queue.put(frame, timeout=1)
            except:
                pass
        
        time.sleep(frame_delay)
    
    print("[Stream Generator] Finished generating frames")


def start_test_stream_server(port=8000, num_frames=1000, fps=10):
    """
    Start local HTTP test stream server.
    
    Parameters
    ----------
    port : int
        Port to serve on (default 8000)
    num_frames : int
        Number of frames to generate
    fps : int
        Frames per second
    
    Returns
    -------
    tuple
        (server, generator_thread) - for cleanup
    """
    frame_queue = queue.Queue(maxsize=2)
    StreamHandler.frame_queue = frame_queue
    
    # Start frame generation thread
    gen_thread = threading.Thread(
        target=generate_test_stream,
        args=(frame_queue, num_frames, fps),
        daemon=True,
        name="StreamGenerator"
    )
    gen_thread.start()
    
    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, StreamHandler)
    
    server_thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
        name="HTTPServer"
    )
    server_thread.start()
    
    print(f"\n{'='*60}")
    print(f"✅ Test Stream Server Started!")
    print(f"{'='*60}")
    print(f"\n📡 Stream URL: http://localhost:{port}/stream.mjpeg")
    print(f"\n⏱️  Serving {num_frames} frames @ {fps} FPS")
    print(f"   Duration: ~{num_frames / fps / 60:.1f} minutes")
    print(f"\n✅ Use this URL in the Live Stream Monitor!")
    print(f"\nPress Ctrl+C to stop...\n")
    
    return httpd, gen_thread


if __name__ == "__main__":
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    httpd, gen_thread = start_test_stream_server(port, num_frames, fps)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[Server] Shutting down...")
        httpd.shutdown()
        print("✅ Server stopped")
