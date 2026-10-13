"""SDK: record agent runs.

>>> from agent_doctor import instrument, trace_run, trace_tool
>>> client = instrument(InferenceClient(model="Qwen/Qwen2.5-7B-Instruct"))
>>> @trace_tool
... def search_orders(order_id: str): ...
>>> @trace_run(agent="support-bot")
... def handle(message: str): ...
"""

from agent_doctor.sdk.decorators import trace_run, trace_tool
from agent_doctor.sdk.instrument import instrument
from agent_doctor.sdk.recorder import RunRecorder, current_run, span

record_run = RunRecorder

__all__ = ["RunRecorder", "current_run", "instrument", "record_run", "span", "trace_run", "trace_tool"]
