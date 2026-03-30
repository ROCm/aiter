"""Notification system with approval gates for the tuning agent."""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import List, Optional


class NotificationLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    APPROVAL = 4


@dataclass
class Notification:
    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class ApprovalRequest:
    question: str
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    approved: Optional[bool] = None


class Notifier:
    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        self.history: List[Notification] = []
        self.pending_approvals: List[ApprovalRequest] = []

    def notify(self, level: NotificationLevel, title: str, message: str) -> Notification:
        notification = Notification(level=level, title=title, message=message)
        self.history.append(notification)
        self._deliver(notification)
        return notification

    def _deliver(self, notification: Notification) -> None:
        prefix = f"[{notification.level.name}]"
        line = f"{prefix} {notification.title}: {notification.message}"
        print(line, file=sys.stderr)
        if notification.level >= NotificationLevel.CRITICAL:
            print("\a", end="", file=sys.stderr, flush=True)

    def request_approval(self, question: str, details: Optional[str] = None) -> bool:
        if self.auto_approve:
            return True

        print(f"\nApproval required: {question}", file=sys.stderr)
        if details:
            print(f"Details: {details}", file=sys.stderr)

        while True:
            response = input("Approve? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            else:
                print("Please enter 'y' or 'n'.", file=sys.stderr)

    def request_approval_async(self, question: str, details: Optional[str] = None) -> None:
        approval_request = ApprovalRequest(question=question, details=details)
        self.pending_approvals.append(approval_request)
