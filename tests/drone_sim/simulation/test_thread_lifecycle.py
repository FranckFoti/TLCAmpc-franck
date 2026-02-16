"""Tests for ThreadLifecycleManager.

Thread lifecycle management for drone threads with spawn, monitor, and graceful shutdown.
Includes integration tests with ThreadSafeMailbox to verify concurrent communication.
"""

import time
import threading
import pytest

from drone_sim.simulation.distributed.thread_lifecycle import ThreadLifecycleManager
from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage
import numpy as np


class TestThreadLifecycleManager:
    """Test ThreadLifecycleManager basic lifecycle operations."""

    def test_spawn_creates_running_thread(self):
        """spawn() creates a thread that runs target function and is_alive()."""
        manager = ThreadLifecycleManager()
        ran = []

        def target(stop_event):
            ran.append(True)
            stop_event.wait(timeout=1.0)

        thread = manager.spawn("drone-1", target)
        time.sleep(0.1)  # Give thread time to start

        assert thread.is_alive()
        assert thread.name == "Drone-drone-1"
        assert len(ran) == 1

        manager.shutdown(timeout=2.0)

    def test_spawn_multiple_threads(self):
        """spawn() can create multiple threads with distinct names."""
        manager = ThreadLifecycleManager()
        threads = []

        def target(stop_event):
            stop_event.wait(timeout=2.0)

        for i in range(4):
            thread = manager.spawn(f"drone-{i}", target)
            threads.append(thread)

        time.sleep(0.1)  # Give threads time to start

        # All threads running
        for thread in threads:
            assert thread.is_alive()

        # All have distinct names
        names = {thread.name for thread in threads}
        assert names == {"Drone-drone-0", "Drone-drone-1", "Drone-drone-2", "Drone-drone-3"}

        manager.shutdown(timeout=2.0)

    def test_shutdown_stops_all_threads(self):
        """shutdown() causes all threads to stop (is_alive() returns False)."""
        manager = ThreadLifecycleManager()

        def target(stop_event):
            while not stop_event.is_set():
                stop_event.wait(timeout=0.1)

        # Spawn 3 threads
        manager.spawn("drone-1", target)
        manager.spawn("drone-2", target)
        manager.spawn("drone-3", target)
        time.sleep(0.1)

        # Shutdown with reasonable timeout
        status = manager.shutdown(timeout=2.0)

        # All threads should be stopped
        assert status == {"drone-1": False, "drone-2": False, "drone-3": False}

    def test_shutdown_sets_stop_event(self):
        """shutdown() sets the shared stop_event."""
        manager = ThreadLifecycleManager()

        def target(stop_event):
            stop_event.wait(timeout=2.0)

        manager.spawn("drone-1", target)
        assert not manager.stop_event.is_set()

        manager.shutdown(timeout=2.0)
        assert manager.stop_event.is_set()

    def test_thread_checks_stop_event(self):
        """Target function that loops checking stop_event exits within timeout after shutdown."""
        manager = ThreadLifecycleManager()
        iterations = []

        def target(stop_event):
            while not stop_event.is_set():
                iterations.append(1)
                stop_event.wait(timeout=0.05)

        manager.spawn("drone-1", target)
        time.sleep(0.2)  # Let it iterate a few times

        initial_count = len(iterations)
        assert initial_count > 0

        manager.shutdown(timeout=1.0)

        # Should not iterate more after shutdown
        time.sleep(0.15)
        assert len(iterations) <= initial_count + 2  # At most 1-2 more due to race

    def test_health_check_reports_alive_threads(self):
        """status() returns dict mapping thread names to alive/dead status."""
        manager = ThreadLifecycleManager()

        def target(stop_event):
            stop_event.wait(timeout=2.0)

        manager.spawn("drone-1", target)
        manager.spawn("drone-2", target)
        time.sleep(0.1)

        status = manager.status()
        assert status == {"drone-1": True, "drone-2": True}

        manager.shutdown(timeout=2.0)
        status = manager.status()
        assert status == {"drone-1": False, "drone-2": False}

    def test_health_check_detects_crashed_thread(self):
        """Thread that raises exception is detected as dead in status()."""
        manager = ThreadLifecycleManager()

        def target(stop_event):
            raise RuntimeError("Simulated crash")

        manager.spawn("drone-1", target)
        time.sleep(0.1)  # Give it time to crash

        status = manager.status()
        assert status == {"drone-1": False}

        manager.shutdown(timeout=1.0)

    def test_stop_event_shared_across_threads(self):
        """All spawned threads share the same stop_event."""
        manager = ThreadLifecycleManager()
        events = []

        def target(stop_event):
            events.append(stop_event)
            stop_event.wait(timeout=2.0)

        manager.spawn("drone-1", target)
        manager.spawn("drone-2", target)
        time.sleep(0.1)

        # All threads received the same event object
        assert len(events) == 2
        assert events[0] is events[1]
        assert events[0] is manager.stop_event

        manager.shutdown(timeout=2.0)

    def test_shutdown_timeout_handles_stuck_thread(self):
        """Thread that ignores stop_event - shutdown returns, thread reported as still alive."""
        manager = ThreadLifecycleManager()

        def target(stop_event):
            # Ignore stop_event and sleep longer than timeout
            time.sleep(5.0)

        manager.spawn("drone-1", target)
        time.sleep(0.1)

        # Shutdown with short timeout
        status = manager.shutdown(timeout=0.5)

        # Thread should still be alive (didn't respect stop_event)
        assert status == {"drone-1": True}

    def test_double_shutdown_is_safe(self):
        """Calling shutdown() twice does not raise."""
        manager = ThreadLifecycleManager()

        def target(stop_event):
            stop_event.wait(timeout=2.0)

        manager.spawn("drone-1", target)
        time.sleep(0.1)

        # First shutdown
        manager.shutdown(timeout=1.0)

        # Second shutdown should not raise
        manager.shutdown(timeout=1.0)


class TestThreadLifecycleIntegration:
    """Integration tests verifying ThreadLifecycleManager works with ThreadSafeMailbox."""

    def test_threads_communicate_via_mailbox(self):
        """Spawn 2 threads via manager; thread A broadcasts, thread B receives."""
        manager = ThreadLifecycleManager()
        mailbox = ThreadSafeMailbox()
        results = {}

        def sender_thread(stop_event, drone_id, target_id):
            # Wait a bit to ensure receiver is waiting
            time.sleep(0.2)
            msg = TrajectoryMessage(
                drone_id=drone_id,
                trajectory=np.array([[1.0, 0.0, 0.0]]),
                predicted_velocities=None,
                timestamp=0,
            )
            mailbox.broadcast(drone_id, msg, {target_id})
            results["sent"] = True
            stop_event.wait(timeout=2.0)

        def receiver_thread(stop_event, drone_id):
            # Wait for message
            notified = mailbox.wait_for_update(drone_id, timeout=1.0)
            if notified:
                messages = mailbox.receive_latest(drone_id)
                results["received"] = len(messages)
            stop_event.wait(timeout=2.0)

        # Start receiver first so it's waiting
        manager.spawn("receiver", receiver_thread, args=("drone-B",))
        # Then start sender after a small delay
        time.sleep(0.05)
        manager.spawn("sender", sender_thread, args=("drone-A", "drone-B"))

        time.sleep(0.6)  # Let communication happen

        assert results.get("sent") is True
        assert results.get("received") == 1

        manager.shutdown(timeout=2.0)

    def test_shutdown_while_threads_use_mailbox(self):
        """Threads actively using mailbox - shutdown stops them cleanly within timeout."""
        manager = ThreadLifecycleManager()
        mailbox = ThreadSafeMailbox()
        broadcast_count = []

        def broadcaster(stop_event, drone_id):
            iteration = 0
            while not stop_event.is_set():
                msg = TrajectoryMessage(
                    drone_id=drone_id,
                    trajectory=np.array([[1.0, 0.0, 0.0]]),
                    predicted_velocities=None,
                    timestamp=iteration,
                )
                mailbox.broadcast(drone_id, msg, {"other-drone"})
                broadcast_count.append(1)
                iteration += 1
                stop_event.wait(timeout=0.05)

        manager.spawn("drone-1", broadcaster, args=("drone-1",))
        manager.spawn("drone-2", broadcaster, args=("drone-2",))

        time.sleep(0.3)  # Let them broadcast a few times
        initial_count = len(broadcast_count)
        assert initial_count > 0

        # Shutdown should complete within timeout
        status = manager.shutdown(timeout=2.0)
        assert status == {"drone-1": False, "drone-2": False}

    def test_four_drone_threads_with_mailbox(self):
        """Spawn 4 threads each broadcasting to neighbors - all messages arrive, clean shutdown."""
        manager = ThreadLifecycleManager()
        mailbox = ThreadSafeMailbox()
        received_counts = {}

        def drone_thread(stop_event, drone_id, neighbors):
            # Broadcast to neighbors
            msg = TrajectoryMessage(
                drone_id=drone_id,
                trajectory=np.array([[float(drone_id[-1]), 0.0, 0.0]]),
                predicted_velocities=None,
                timestamp=0,
            )
            mailbox.broadcast(drone_id, msg, neighbors)

            # Wait for messages from neighbors
            time.sleep(0.2)  # Give time for messages to arrive
            messages = mailbox.receive_latest(drone_id)
            received_counts[drone_id] = len(messages)

            stop_event.wait(timeout=2.0)

        # 4-drone ring topology: each drone has 2 neighbors
        manager.spawn("drone-0", drone_thread, args=("drone-0", {"drone-1", "drone-3"}))
        manager.spawn("drone-1", drone_thread, args=("drone-1", {"drone-0", "drone-2"}))
        manager.spawn("drone-2", drone_thread, args=("drone-2", {"drone-1", "drone-3"}))
        manager.spawn("drone-3", drone_thread, args=("drone-3", {"drone-0", "drone-2"}))

        time.sleep(0.5)  # Let all communication complete

        # Each drone should receive messages from its 2 neighbors
        assert received_counts == {
            "drone-0": 2,
            "drone-1": 2,
            "drone-2": 2,
            "drone-3": 2,
        }

        # Clean shutdown
        status = manager.shutdown(timeout=2.0)
        assert all(not alive for alive in status.values())
