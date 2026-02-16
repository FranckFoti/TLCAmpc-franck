"""Thread lifecycle management for drone threads.

Provides spawn, health monitoring, and graceful shutdown with timeout.
All spawned threads share a single stop_event for cooperative shutdown.
"""

from __future__ import annotations

import threading
from typing import Callable


class ThreadLifecycleManager:
    """Manages thread lifecycle for drone threads.

    Provides spawn, health monitoring, and graceful shutdown with timeout.
    All spawned threads share a single stop_event for cooperative shutdown.

    Thread safety:
    - All methods are thread-safe
    - Uses threading.RLock to protect _threads dict
    - Shared threading.Event for cooperative shutdown signal

    Target function requirements:
    - Must accept stop_event as first argument
    - Should check stop_event.is_set() in their loop
    - Should use stop_event.wait(timeout=...) instead of time.sleep() for responsive shutdown
    """

    def __init__(self) -> None:
        """Initialize thread lifecycle manager.

        Creates:
        - Shared stop_event for cooperative shutdown
        - Empty threads dict for tracking spawned threads
        - Lock for thread-safe access to _threads dict
        """
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._threads: dict[str, threading.Thread] = {}

    @property
    def stop_event(self) -> threading.Event:
        """Get the shared stop event.

        All spawned threads share this event for cooperative shutdown.
        External code can check this event if needed.
        """
        return self._stop_event

    def spawn(
        self,
        thread_id: str,
        target: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> threading.Thread:
        """Spawn a new thread with given target function.

        The target function MUST accept stop_event as its first argument.
        Thread is started immediately and stored in _threads dict.

        :param thread_id: Unique identifier for the thread (used in thread name and status dict)
        :param target: Callable to execute in the thread
        :param args: Additional positional arguments to pass to target (after stop_event)
        :param kwargs: Keyword arguments to pass to target
        :return: The created and started Thread object
        """
        if kwargs is None:
            kwargs = {}

        # Create thread with name for debuggability
        # Pass stop_event as first argument to target
        thread = threading.Thread(
            name=f"Drone-{thread_id}",
            target=target,
            args=(self._stop_event, *args),
            kwargs=kwargs,
            daemon=False,  # Explicit shutdown, no zombie threads
        )

        with self._lock:
            self._threads[thread_id] = thread

        thread.start()
        return thread

    def status(self) -> dict[str, bool]:
        """Get health status of all spawned threads.

        :return: Dict mapping thread_id to is_alive() status (True = running, False = stopped/crashed)
        """
        with self._lock:
            return {thread_id: thread.is_alive() for thread_id, thread in self._threads.items()}

    def shutdown(self, timeout: float = 5.0) -> dict[str, bool]:
        """Gracefully shut down all threads.

        Sets the shared stop_event, then joins each thread with timeout.
        Threads that check stop_event.is_set() will terminate within timeout.
        Threads that ignore stop_event will still be alive after timeout expires.

        :param timeout: Maximum time to wait for all threads to stop (divided across threads)
        :return: Status dict (which threads are still alive after shutdown)
        """
        # Set stop signal
        self._stop_event.set()

        with self._lock:
            thread_list = list(self._threads.values())

        # Divide timeout across all threads
        if thread_list:
            per_thread_timeout = timeout / len(thread_list)
        else:
            per_thread_timeout = timeout

        # Join each thread with timeout
        for thread in thread_list:
            if thread.is_alive():
                thread.join(timeout=per_thread_timeout)

        # Return final status
        return self.status()
