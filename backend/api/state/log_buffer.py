import threading
from collections import deque
from typing import List
import os
import asyncio

LOG_FILE = os.environ.get('TRAINING_LOG_FILE', '/data/logs/training_console.log')

class LogBuffer:
    def __init__(self, maxlines: int = 500):
        self._lines: deque = deque(maxlen=maxlines)
        self._lock = threading.Lock()
        # Changed: store (loop, queue) tuples instead of bare queues
        self._subscribers: list = []
        self._log_file = LOG_FILE
        self._load_from_disk()

    def write(self, text: str):
        for line in text.splitlines(keepends=True):
            with self._lock:
                self._lines.append(line)
                subscribers_snapshot = list(self._subscribers)

            # Persist to disk
            try:
                os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
                with open(self._log_file, 'a', errors='replace') as f:
                    f.write(line)
            except Exception:
                pass

            # ── KEY FIX: use call_soon_threadsafe so the put_nowait
            #    runs inside the event loop's thread, not the training thread ──
            for (loop, q) in subscribers_snapshot:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, line)
                except Exception:
                    pass

    def get_history(self) -> List[str]:
        with self._lock:
            return list(self._lines)

    def subscribe(self, queue):
        # Capture the running loop at subscription time (called from async context)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        with self._lock:
            self._subscribers.append((loop, queue))

    def unsubscribe(self, queue):
        with self._lock:
            self._subscribers = [(l, q) for (l, q) in self._subscribers if q is not queue]

    def clear(self):
        with self._lock:
            self._lines.clear()
        try:
            open(self._log_file, 'w').close()
        except Exception:
            pass

    def _load_from_disk(self):
        try:
            if os.path.exists(self._log_file):
                with open(self._log_file, 'r', errors='replace') as f:
                    all_lines = f.readlines()
                    for line in all_lines[-self._lines.maxlen:]:
                        self._lines.append(line)
        except Exception:
            pass

# Singleton
log_buffer = LogBuffer()
