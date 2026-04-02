"""
HTTP sender with offline buffering.
If Django is unreachable, payloads are queued and sent when it comes back.
"""
import time
import requests
from datetime import datetime
from config import BACKEND_URL, SEND_INTERVAL
 
 
class DataSender:
 
    def __init__(self):
        self.last_send_time = 0
        self.buffer = []
 
    def maybe_send(self, data: dict) -> bool:
        """Send data if SEND_INTERVAL has elapsed. Returns True if sent."""
        now = time.time()
        if now - self.last_send_time < SEND_INTERVAL:
            return False
 
        payload = {
            'timestamp': datetime.now().isoformat(),
            'people_count': data['people_count'],
            'estimated_wait_seconds': data['estimated_wait_sec'],
            'service_rate': data['service_rate'],
        }
        self.buffer.append(payload)
 
        try:
            for item in self.buffer:
                resp = requests.post(BACKEND_URL, json=item, timeout=5)
                resp.raise_for_status()
            sent_count = len(self.buffer)
            self.buffer.clear()
            self.last_send_time = now
            print(f'[SEND] OK - {data["people_count"]} people, '
                  f'{data["estimated_wait_min"]} min wait'
                  f'{" (flushed " + str(sent_count) + " buffered)" if sent_count > 1 else ""}')
            return True
        except requests.exceptions.RequestException as e:
            print(f'[SEND] Failed (buffered {len(self.buffer)}): {e}')
            self.last_send_time = now
            return False
