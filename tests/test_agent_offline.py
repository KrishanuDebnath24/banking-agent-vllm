"""Offline test: real agent graph + real tools, fake LLM and fake OBP bank."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents.agent import build_agent
from agents.context import CustomerContext


class ToolCapableFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tx(date, amount, desc):
    return {"details": {"completed": f"{date}T10:00:00Z", "description": desc,
                        "value": {"currency": "GBP", "amount": f"{amount:.2f}"},
                        "new_balance": {"currency": "GBP", "amount": "0"}}}


class FakeBank:
    calls = []
    def balance_amount(self, token, bank, acc):
        self.calls.append(("balance", acc)); return ({"cur1": 2500.12, "sav1": 300.0}[acc], "GBP")
    def transactions(self, token, bank, acc):
        self.calls.append(("tx", acc))
        return [tx("2019-08-19", -232, "Description abc"),          # old sandbox noise: hidden
                tx("2026-10-12", 712.50, "ACME Analytics Ltd - Weekly pay"),
                tx("2026-10-12", -41.87, "Tesco Stores 2041"),
                tx("2026-10-12", -48.03, "Tesco Stores 2041"),
                tx("2026-10-12", -36.54, "Sainsbury's 0612"),
                tx("2026-10-12", -795.00, "Rent - Harbour Lettings")]


ctx = CustomerContext(label="A", username="robert", token="secret-token", bank_id="gh.29.uk", currency="GBP",
                      accounts={"current": "cur1", "savings": "sav1"}, client=FakeBank(), show_since="2026-10-11")

script = [
    AIMessage(content="", tool_calls=[{"name": "list_my_accounts", "args": {}, "id": "1"}]),
    AIMessage(content="", tool_calls=[{"name": "summarize_spending", "args": {"account": "current", "search": "tesco,sainsbury"}, "id": "2"}]),
    AIMessage(content="", tool_calls=[{"name": "get_transactions", "args": {"account": "someone-elses-account"}, "id": "3"}]),
    AIMessage(content="Done."),
]
agent = build_agent(model=ToolCapableFake(messages=iter(script)))
out = agent.invoke({"messages": [{"role": "user", "content": "hi"}]},
                   config={"configurable": {"thread_id": "t1"}}, context=ctx)

tool_msgs = [m for m in out["messages"] if m.type == "tool"]
accounts = json.loads(tool_msgs[0].content)
spend = json.loads(tool_msgs[1].content)
denied = tool_msgs[2].content
print("accounts:", accounts)
print("grocery spend:", spend["money_out"], spend["spending_by_payee"], "tx counted:", spend["transactions"])
print("denied:", denied)
assert accounts[0]["balance"] == 2500.12
assert spend["money_out"] == round(41.87 + 48.03 + 36.54, 2) and spend["transactions"] == 3
assert denied.startswith("Access denied")
assert not any(c[1] == "someone-elses-account" for c in FakeBank.calls), "denied account must never reach the bank"
assert "secret-token" not in json.dumps([m.content for m in out["messages"]]), "token must not appear in messages"
print("ALL OFFLINE CHECKS PASSED")
