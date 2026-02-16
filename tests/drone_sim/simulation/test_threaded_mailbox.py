"""Tests for thread-safe mailbox system.

Tests for ThreadSafeMailbox and TrajectoryMessage extension with safety_zone_radius.
Following TDD: RED (write failing tests) -> GREEN (implement) -> REFACTOR (clean up).
"""

from __future__ import annotations

import time
import threading
import numpy as np
import pytest
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage
from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox


class TestTrajectoryMessageExtension:
    """Tests for TrajectoryMessage extension with safety_zone_radius field."""

    def test_message_with_safety_zone_radius_float(self):
        """Test TrajectoryMessage accepts safety_zone_radius as float."""
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        msg = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            safety_zone_radius=1.5,
        )
        assert msg.safety_zone_radius == 1.5

    def test_message_with_safety_zone_radius_ndarray(self):
        """Test TrajectoryMessage accepts safety_zone_radius as np.ndarray (per-step radii)."""
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        radii = np.array([1.0, 1.2])
        msg = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            safety_zone_radius=radii,
        )
        np.testing.assert_array_equal(msg.safety_zone_radius, radii)

    def test_message_backward_compat_no_safety_zone_radius(self):
        """Test TrajectoryMessage without safety_zone_radius defaults to None (backward compat)."""
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        msg = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
        )
        assert msg.safety_zone_radius is None


class TestThreadSafeMailboxBasic:
    """Basic functionality tests for ThreadSafeMailbox."""

    @pytest.fixture
    def mailbox(self) -> ThreadSafeMailbox:
        """Create an empty ThreadSafeMailbox."""
        return ThreadSafeMailbox()

    def test_broadcast_to_neighbors(self, mailbox: ThreadSafeMailbox):
        """Test broadcast posts message to correct neighbor inboxes."""
        trajectory = np.zeros((5, 3))
        msg = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
        )
        neighbor_ids = {"drone-2", "drone-3"}

        mailbox.broadcast("drone-1", msg, neighbor_ids)

        msgs_drone2 = mailbox.receive_latest("drone-2")
        msgs_drone3 = mailbox.receive_latest("drone-3")
        msgs_drone4 = mailbox.receive_latest("drone-4")

        assert "drone-1" in msgs_drone2
        assert "drone-1" in msgs_drone3
        assert "drone-1" not in msgs_drone4  # Not a neighbor

    def test_receive_latest_returns_latest_per_sender(self, mailbox: ThreadSafeMailbox):
        """Test receive_latest returns only most recent message from each sender."""
        trajectory1 = np.ones((5, 3))
        trajectory2 = np.ones((5, 3)) * 2

        msg1 = TrajectoryMessage("drone-1", trajectory1, None, 0)
        msg2 = TrajectoryMessage("drone-1", trajectory2, None, 1)

        mailbox.broadcast("drone-1", msg1, {"drone-2"})
        mailbox.broadcast("drone-1", msg2, {"drone-2"})

        msgs = mailbox.receive_latest("drone-2")
        assert len(msgs) == 1
        assert "drone-1" in msgs
        np.testing.assert_array_equal(msgs["drone-1"].trajectory, trajectory2)
        assert msgs["drone-1"].timestamp == 1

    def test_receive_latest_returns_empty_for_unknown_drone(self, mailbox: ThreadSafeMailbox):
        """Test receive_latest returns empty dict for drone with no messages."""
        msgs = mailbox.receive_latest("unknown-drone")
        assert msgs == {}
        assert isinstance(msgs, dict)

    def test_receive_latest_nonblocking(self, mailbox: ThreadSafeMailbox):
        """Test receive_latest returns immediately even when inbox empty."""
        start_time = time.time()
        msgs = mailbox.receive_latest("drone-1")
        elapsed = time.time() - start_time

        assert msgs == {}
        assert elapsed < 0.1  # Should be near-instant

    def test_broadcast_does_not_affect_non_neighbors(self, mailbox: ThreadSafeMailbox):
        """Test message only reaches specified neighbor_ids."""
        msg = TrajectoryMessage("drone-1", np.zeros((5, 3)), None, 0)
        mailbox.broadcast("drone-1", msg, {"drone-2"})

        assert "drone-1" in mailbox.receive_latest("drone-2")
        assert "drone-1" not in mailbox.receive_latest("drone-3")
        assert "drone-1" not in mailbox.receive_latest("drone-4")

    def test_clear_removes_all_messages(self, mailbox: ThreadSafeMailbox):
        """Test clear empties all inboxes."""
        msg1 = TrajectoryMessage("drone-1", np.zeros((5, 3)), None, 0)
        msg2 = TrajectoryMessage("drone-2", np.zeros((5, 3)), None, 0)

        mailbox.broadcast("drone-1", msg1, {"drone-2", "drone-3"})
        mailbox.broadcast("drone-2", msg2, {"drone-1", "drone-3"})

        # Verify messages exist
        assert len(mailbox.receive_latest("drone-2")) > 0
        assert len(mailbox.receive_latest("drone-1")) > 0

        mailbox.clear()

        # Verify all inboxes empty
        assert mailbox.receive_latest("drone-1") == {}
        assert mailbox.receive_latest("drone-2") == {}
        assert mailbox.receive_latest("drone-3") == {}

    def test_clear_drone_removes_single_inbox(self, mailbox: ThreadSafeMailbox):
        """Test clear_drone empties one drone's inbox without affecting others."""
        msg = TrajectoryMessage("drone-1", np.zeros((5, 3)), None, 0)
        mailbox.broadcast("drone-1", msg, {"drone-2", "drone-3"})

        mailbox.clear_drone("drone-2")

        assert mailbox.receive_latest("drone-2") == {}
        assert "drone-1" in mailbox.receive_latest("drone-3")


class TestThreadSafeMailboxNotification:
    """Tests for ThreadSafeMailbox event notification and blocking wait."""

    @pytest.fixture
    def mailbox(self) -> ThreadSafeMailbox:
        """Create an empty ThreadSafeMailbox."""
        return ThreadSafeMailbox()

    def test_wait_for_update_blocks_until_message(self, mailbox: ThreadSafeMailbox):
        """Test wait_for_update blocks until a message arrives."""
        received_msg = {}

        def waiter():
            notified = mailbox.wait_for_update("drone-2", timeout=2.0)
            assert notified is True
            msgs = mailbox.receive_latest("drone-2")
            received_msg.update(msgs)

        # Start waiter thread
        waiter_thread = threading.Thread(target=waiter)
        waiter_thread.start()

        # Give waiter time to start waiting
        time.sleep(0.1)

        # Broadcast from main thread
        msg = TrajectoryMessage("drone-1", np.zeros((5, 3)), None, 0)
        mailbox.broadcast("drone-1", msg, {"drone-2"})

        waiter_thread.join(timeout=3.0)
        assert not waiter_thread.is_alive()
        assert "drone-1" in received_msg

    def test_wait_for_update_timeout_returns_empty(self, mailbox: ThreadSafeMailbox):
        """Test wait_for_update returns False if no message arrives before timeout."""
        start_time = time.time()
        notified = mailbox.wait_for_update("drone-1", timeout=0.2)
        elapsed = time.time() - start_time

        assert notified is False
        assert 0.15 < elapsed < 0.4  # Should timeout around 0.2s

    def test_subscriber_notified_on_broadcast(self, mailbox: ThreadSafeMailbox):
        """Test broadcast triggers notification for waiting subscriber."""
        notification_count = [0]

        def subscriber():
            for _ in range(3):
                notified = mailbox.wait_for_update("drone-2", timeout=1.0)
                if notified:
                    notification_count[0] += 1

        subscriber_thread = threading.Thread(target=subscriber)
        subscriber_thread.start()

        # Send 3 messages
        for i in range(3):
            time.sleep(0.1)
            msg = TrajectoryMessage("drone-1", np.zeros((5, 3)), None, i)
            mailbox.broadcast("drone-1", msg, {"drone-2"})

        subscriber_thread.join(timeout=5.0)
        assert notification_count[0] == 3


class TestThreadSafeMailboxConcurrency:
    """Thread safety tests with concurrent operations."""

    @pytest.fixture
    def mailbox(self) -> ThreadSafeMailbox:
        """Create an empty ThreadSafeMailbox."""
        return ThreadSafeMailbox()

    def test_concurrent_broadcasts_no_data_loss(self, mailbox: ThreadSafeMailbox):
        """Test 4 threads each broadcast 10 messages - all messages arrive correctly."""
        num_senders = 4
        messages_per_sender = 10
        receiver_id = "receiver"

        def broadcaster(sender_id: str):
            for i in range(messages_per_sender):
                trajectory = np.ones((5, 3)) * (int(sender_id.split("-")[1]) * 100 + i)
                msg = TrajectoryMessage(sender_id, trajectory, None, i)
                mailbox.broadcast(sender_id, msg, {receiver_id})
                time.sleep(0.001)  # Small delay to interleave

        threads = []
        for i in range(num_senders):
            sender_id = f"sender-{i}"
            thread = threading.Thread(target=broadcaster, args=(sender_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=5.0)

        # Verify: receiver should have latest message from each sender
        msgs = mailbox.receive_latest(receiver_id)
        assert len(msgs) == num_senders

        for i in range(num_senders):
            sender_id = f"sender-{i}"
            assert sender_id in msgs
            # Last message should have timestamp = messages_per_sender - 1
            assert msgs[sender_id].timestamp == messages_per_sender - 1

    def test_concurrent_read_write(self, mailbox: ThreadSafeMailbox):
        """Test 2 writers + 2 readers running simultaneously - no exceptions, readers get valid data."""
        errors = []
        num_iterations = 50

        def writer(sender_id: str):
            try:
                for i in range(num_iterations):
                    msg = TrajectoryMessage(sender_id, np.zeros((5, 3)), None, i)
                    mailbox.broadcast(sender_id, msg, {"reader-1", "reader-2"})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Writer {sender_id}: {e}")

        def reader(reader_id: str):
            try:
                for _ in range(num_iterations):
                    msgs = mailbox.receive_latest(reader_id)
                    # Validate all messages have correct structure
                    for sender_id, msg in msgs.items():
                        assert isinstance(msg, TrajectoryMessage)
                        assert msg.trajectory.shape == (5, 3)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader {reader_id}: {e}")

        threads = []
        for i in range(2):
            threads.append(threading.Thread(target=writer, args=(f"writer-{i}",)))
            threads.append(threading.Thread(target=reader, args=(f"reader-{i}",)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=10.0)

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_broadcast_and_wait_concurrent(self, mailbox: ThreadSafeMailbox):
        """Test one thread waits while another broadcasts - waiter receives message."""
        received = {}

        def waiter():
            notified = mailbox.wait_for_update("drone-2", timeout=2.0)
            if notified:
                msgs = mailbox.receive_latest("drone-2")
                received.update(msgs)

        waiter_thread = threading.Thread(target=waiter)
        waiter_thread.start()

        time.sleep(0.1)  # Ensure waiter is waiting

        # Broadcast from main thread
        msg = TrajectoryMessage("drone-1", np.ones((5, 3)), None, 42)
        mailbox.broadcast("drone-1", msg, {"drone-2"})

        waiter_thread.join(timeout=3.0)
        assert "drone-1" in received
        assert received["drone-1"].timestamp == 42
