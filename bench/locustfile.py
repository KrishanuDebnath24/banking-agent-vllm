"""E7: how many customers the whole app serves, measured through the web API like the real page.

Each simulated user signs in as a visitor (their own pretend bank), then has a scripted banking
conversation: send a message, check back until the answer is ready, read for 10-30 seconds, send
the next. "agent turn" in the results is the time from sending a message to the finished answer,
every LLM and tool call included.

    pip install -r bench/requirements.txt
    locust -f bench/locustfile.py --host http://127.0.0.1:8000 --headless \
           --users 50 --spawn-rate 1 --run-time 15m --csv bench/results/e7-50users

The app under test needs a test-only set-up, or its visitor limits stop the load test:
    AUTH_MODE=dev                     (visitors sign in without an emailed code)
    GUEST_SIGNUPS_PER_IP_HOUR=10000   GUEST_DAILY_MESSAGES=10000   DAILY_SPEND_CAP_USD=1000
Never use these settings on a public site.

Write actions (transfers, address changes) are declined at the confirmation step, so nothing
changes in the pretend banks. Settings: LOCUST_THINK_MIN / LOCUST_THINK_MAX (seconds, default 10 / 30),
LOCUST_TURN_TIMEOUT (seconds, default 180).
"""

from __future__ import annotations

import os
import random
import time
import uuid

from locust import HttpUser, between, events, task

THINK_MIN = float(os.environ.get("LOCUST_THINK_MIN", "10"))
THINK_MAX = float(os.environ.get("LOCUST_THINK_MAX", "30"))
TURN_TIMEOUT = float(os.environ.get("LOCUST_TURN_TIMEOUT", "180"))

CONVERSATIONS = [
    ["What are my balances?", "How much did I spend on groceries last month?", "Which shop was the biggest?",
     "Show me a chart of my spending by category this month.", "Thanks!"],
    ["Show my last five transactions.", "Did I pay rent this month?", "What subscriptions do I pay for?",
     "How much did I spend on eating out in the last 30 days?", "Is that more than the month before?"],
    ["What's in my savings?", "Move £50 from current to savings.", "What's my current balance now?",
     "Show my spending by month as a line chart.", "Great, that's all."],
]


class Customer(HttpUser):
    wait_time = between(THINK_MIN, THINK_MAX)

    def on_start(self) -> None:
        n = uuid.uuid4().hex[:10]
        r = self.client.post("/api/login", json={"email": f"load-{n}@example.com", "name": f"Load {n[:4]}"},
                             name="sign in")
        if r.status_code != 200:
            raise RuntimeError(f"sign-in failed ({r.status_code}): set AUTH_MODE=dev on the app under test")
        self.script = list(random.choice(CONVERSATIONS))
        self.thread_id: str | None = None

    @task
    def say_next(self) -> None:
        if not self.script:  # conversation finished: start a new one
            self.script, self.thread_id = list(random.choice(CONVERSATIONS)), None
        self.turn(self.script.pop(0))

    def turn(self, message: str) -> None:
        started = time.perf_counter()
        body = {"message": message, **({"thread_id": self.thread_id} if self.thread_id else {})}
        r = self.client.post("/api/chat", json=body, name="send message")
        if r.status_code != 202:
            self._record(started, f"send failed: HTTP {r.status_code}")
            return
        data = r.json()
        self.thread_id, turn_id, seen = data["thread_id"], data["turn_id"], 0
        while time.perf_counter() - started < TURN_TIMEOUT:
            r = self.client.get(f"/api/turns/{turn_id}?wait=20&seen={seen}", name="check back")
            if r.status_code != 200:
                self._record(started, f"check back failed: HTTP {r.status_code}")
                return
            data = r.json()
            seen = len(data.get("steps") or [])
            if data["status"] == "running":
                continue
            if data["status"] == "needs_confirmation":  # decline: nothing changes in the pretend bank
                r = self.client.post("/api/confirm", json={"thread_id": self.thread_id, "approve": False},
                                     name="decline action")
                if r.status_code != 202:
                    self._record(started, f"decline failed: HTTP {r.status_code}")
                    return
                turn_id, seen = r.json()["turn_id"], 0   # the app finishes the answer in a new turn
                continue
            error = data.get("error")
            self._record(started, None if data["status"] == "answered" and not error
                         else f"{data['status']}: {error or 'no answer'}")
            return
        self._record(started, "timed out")

    def _record(self, started: float, error: str | None) -> None:
        events.request.fire(request_type="TURN", name="agent turn",
                            response_time=(time.perf_counter() - started) * 1000, response_length=0,
                            exception=RuntimeError(error) if error else None, context={})
