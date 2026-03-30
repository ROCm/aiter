"""Tests for the notification system."""

import sys
from datetime import datetime
from unittest.mock import patch

import pytest

from aiter.ops.triton.utils._triton.tunning.tuning_agent.notifications import (
    ApprovalRequest,
    Notification,
    NotificationLevel,
    Notifier,
)


class TestNotificationLevel:
    def test_ordering_critical_greater_than_info(self):
        assert NotificationLevel.CRITICAL > NotificationLevel.INFO

    def test_ordering_approval_greater_than_critical(self):
        assert NotificationLevel.APPROVAL > NotificationLevel.CRITICAL

    def test_ordering_full_sequence(self):
        levels = [
            NotificationLevel.DEBUG,
            NotificationLevel.INFO,
            NotificationLevel.WARNING,
            NotificationLevel.CRITICAL,
            NotificationLevel.APPROVAL,
        ]
        for i in range(len(levels) - 1):
            assert levels[i] < levels[i + 1]

    def test_values(self):
        assert NotificationLevel.DEBUG == 0
        assert NotificationLevel.INFO == 1
        assert NotificationLevel.WARNING == 2
        assert NotificationLevel.CRITICAL == 3
        assert NotificationLevel.APPROVAL == 4


class TestNotification:
    def test_creation_with_required_fields(self):
        n = Notification(
            level=NotificationLevel.INFO,
            title="Test Title",
            message="Test message",
        )
        assert n.level == NotificationLevel.INFO
        assert n.title == "Test Title"
        assert n.message == "Test message"

    def test_default_timestamp_is_recent(self):
        before = datetime.now()
        n = Notification(
            level=NotificationLevel.DEBUG,
            title="T",
            message="M",
        )
        after = datetime.now()
        assert before <= n.timestamp <= after

    def test_default_acknowledged_is_false(self):
        n = Notification(level=NotificationLevel.WARNING, title="T", message="M")
        assert n.acknowledged is False

    def test_acknowledged_can_be_set(self):
        n = Notification(level=NotificationLevel.INFO, title="T", message="M", acknowledged=True)
        assert n.acknowledged is True

    def test_explicit_timestamp(self):
        ts = datetime(2025, 1, 15, 10, 30, 0)
        n = Notification(level=NotificationLevel.INFO, title="T", message="M", timestamp=ts)
        assert n.timestamp == ts


class TestApprovalRequest:
    def test_creation_with_question(self):
        ar = ApprovalRequest(question="Proceed?")
        assert ar.question == "Proceed?"
        assert ar.details is None
        assert ar.approved is None

    def test_creation_with_details(self):
        ar = ApprovalRequest(question="Delete files?", details="Will remove 50 files")
        assert ar.details == "Will remove 50 files"

    def test_default_timestamp_is_recent(self):
        before = datetime.now()
        ar = ApprovalRequest(question="Q")
        after = datetime.now()
        assert before <= ar.timestamp <= after

    def test_approved_can_be_set(self):
        ar = ApprovalRequest(question="Q", approved=True)
        assert ar.approved is True

        ar2 = ApprovalRequest(question="Q", approved=False)
        assert ar2.approved is False


class TestNotifier:
    def test_initial_state(self):
        notifier = Notifier()
        assert notifier.history == []
        assert notifier.pending_approvals == []
        assert notifier.auto_approve is False

    def test_auto_approve_flag(self):
        notifier = Notifier(auto_approve=True)
        assert notifier.auto_approve is True

    def test_notify_records_history(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.INFO, "Title", "Message")
        assert len(notifier.history) == 1
        assert notifier.history[0].level == NotificationLevel.INFO
        assert notifier.history[0].title == "Title"
        assert notifier.history[0].message == "Message"

    def test_notify_multiple_records_all(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.DEBUG, "T1", "M1")
        notifier.notify(NotificationLevel.WARNING, "T2", "M2")
        notifier.notify(NotificationLevel.CRITICAL, "T3", "M3")
        assert len(notifier.history) == 3

    def test_notify_prints_to_stderr(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.INFO, "MyTitle", "MyMessage")
        captured = capsys.readouterr()
        assert "[INFO]" in captured.err
        assert "MyTitle" in captured.err
        assert "MyMessage" in captured.err

    def test_notify_critical_prints_bell(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.CRITICAL, "Alert", "Something critical")
        captured = capsys.readouterr()
        assert "\a" in captured.err

    def test_notify_approval_level_prints_bell(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.APPROVAL, "Approval", "Need approval")
        captured = capsys.readouterr()
        assert "\a" in captured.err

    def test_notify_info_no_bell(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.INFO, "Info", "Just info")
        captured = capsys.readouterr()
        assert "\a" not in captured.err

    def test_notify_warning_no_bell(self, capsys):
        notifier = Notifier()
        notifier.notify(NotificationLevel.WARNING, "Warn", "A warning")
        captured = capsys.readouterr()
        assert "\a" not in captured.err

    def test_notify_returns_notification(self, capsys):
        notifier = Notifier()
        result = notifier.notify(NotificationLevel.INFO, "T", "M")
        assert isinstance(result, Notification)
        assert result.title == "T"

    def test_auto_approve_returns_true(self):
        notifier = Notifier(auto_approve=True)
        result = notifier.request_approval("Should we proceed?")
        assert result is True

    def test_auto_approve_with_details_returns_true(self):
        notifier = Notifier(auto_approve=True)
        result = notifier.request_approval("Proceed?", details="Some details")
        assert result is True

    def test_request_approval_yes(self, capsys):
        notifier = Notifier(auto_approve=False)
        with patch("builtins.input", return_value="y"):
            result = notifier.request_approval("Continue?")
        assert result is True

    def test_request_approval_no(self, capsys):
        notifier = Notifier(auto_approve=False)
        with patch("builtins.input", return_value="n"):
            result = notifier.request_approval("Continue?")
        assert result is False

    def test_request_approval_yes_full_word(self, capsys):
        notifier = Notifier(auto_approve=False)
        with patch("builtins.input", return_value="yes"):
            result = notifier.request_approval("Continue?")
        assert result is True

    def test_request_approval_no_full_word(self, capsys):
        notifier = Notifier(auto_approve=False)
        with patch("builtins.input", return_value="no"):
            result = notifier.request_approval("Continue?")
        assert result is False

    def test_request_approval_loops_on_invalid_then_accepts(self, capsys):
        notifier = Notifier(auto_approve=False)
        with patch("builtins.input", side_effect=["maybe", "invalid", "y"]):
            result = notifier.request_approval("Continue?")
        assert result is True

    def test_request_approval_async_queues_without_blocking(self):
        notifier = Notifier()
        notifier.request_approval_async("Should we run?")
        assert len(notifier.pending_approvals) == 1
        assert notifier.pending_approvals[0].question == "Should we run?"
        assert notifier.pending_approvals[0].details is None

    def test_request_approval_async_with_details(self):
        notifier = Notifier()
        notifier.request_approval_async("Delete?", details="Will delete 10 items")
        assert len(notifier.pending_approvals) == 1
        assert notifier.pending_approvals[0].details == "Will delete 10 items"

    def test_request_approval_async_multiple(self):
        notifier = Notifier()
        notifier.request_approval_async("Q1")
        notifier.request_approval_async("Q2")
        notifier.request_approval_async("Q3")
        assert len(notifier.pending_approvals) == 3
        assert notifier.pending_approvals[0].question == "Q1"
        assert notifier.pending_approvals[2].question == "Q3"

    def test_pending_approvals_approved_field_is_none(self):
        notifier = Notifier()
        notifier.request_approval_async("Proceed?")
        assert notifier.pending_approvals[0].approved is None
