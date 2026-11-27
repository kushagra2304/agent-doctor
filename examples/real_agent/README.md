# The real agent

A small tool-calling support agent on a real open model. Unlike the [bad agent](../bad_agent/),
nothing is planted: the model decides which tools to call, in what order and how often.
The tools are small in-memory tables (orders, tracking, help centre, stock, and a
`send_reply` side-effect tool), and one is broken on purpose: Ekart's tracking API, used
by order 9012, always times out.

It runs free on your own machine with [Ollama](https://ollama.com/download):

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:1.5b                       # only for the eval's smaller-model variant

export AD_BASE_URL=http://localhost:11434/v1   # PowerShell: $env:AD_BASE_URL = "..."
python -m examples.real_agent.record           # 10 questions, about 15 s on a laptop GPU
agent-doctor --db examples/real_agent/traces.db scan --prices examples/real_agent/local_pricing.yaml
agent-doctor --db examples/real_agent/traces.db eval examples/real_agent/eval_suite.py \
    --prices examples/real_agent/local_pricing.yaml
```

Leave `AD_BASE_URL` unset to use Hugging Face Inference Providers instead (`hf auth login`;
needs credits). `AD_MODEL` picks the model.

## What we found (qwen2.5:3b, November 2026)

Each answer was graded against [questions.py](questions.py), and every finding was checked
by hand against the trace.

**`scan`: 0 findings on 40 runs, and that is correct.** The agent made no repeated calls,
retries or loops. Its tools take no time, its prompts are too short to cache, and the
model is free. There was nothing to waste. Agent Doctor stayed quiet instead of inventing
findings.

**Quality failures belong to the eval.** v1 answered 8 of 10 questions correctly. The
failures were things `scan` is not meant to see:

| Question | What the model did |
|---|---|
| Where is my order #1234? | Looked up the order but never called `get_tracking`, so no delivery day |
| How long does standard shipping take? | Answered from memory without searching the help centre (right by luck) |
| Is SKU-GRN-38 in stock, and how fast is express? | Never looked up express shipping; did not say how many were in stock |
| *(all)* | Called `send_reply` in 2 of 10 runs, although the prompt says "exactly once" |

**The v2 prompt** spells out which tool answers which kind of question.

| Variant | Correct | Looked facts up | Called send_reply once |
|---|---|---|---|
| v1 (recorded) | 80% | 90% | 20% |
| v1 rerun | 80% | 90% | 20% |
| v2 | 80% | **100%** | **0%** |
| v2 on qwen2.5:1.5b | 70% | 60% | 0% |

- **v2 is not better.** It always looks facts up now, and it fixes the express-shipping
  question. But it breaks "Has order 9012 shipped?": it reports that tracking is
  unavailable and never says the order is only packed. It also stopped calling
  `send_reply` entirely. A 3B model can't be prompted into reliably finishing with a
  tool call. That step belongs in code.
- **The 1.5B model guesses.** It answered three policy questions from memory, invented a
  delivery date ("2023-10-15") instead of calling `get_tracking`, and once returned a
  template placeholder (`<Order status>`) as the answer.
- **Temperature 0 is repeatable here, but latency is not.** The v1 rerun gave the same
  answers as the recording, word for word. Its total time was 24% lower on one run and
  27% higher on the next, from model warm-up alone. With 10 cases, ignore time changes
  smaller than that.

## What it found in Agent Doctor

Running on real data turned up four problems, all fixed and covered by tests:

1. **False alarm: "get_tracking never succeeded, but the agent answered anyway".** This was
   flagged on runs whose answer said the tracking information was unavailable or had
   timed out. Telling the customer is the right behaviour, and is no longer flagged. When
   the tool worked for other orders, the title now names the failing arguments.
2. **Comparisons hid regressions.** A case passes only if every scorer passes it, and
   `replied_once` failed nearly everywhere. So the overall verdict said "same" while
   correctness fell from 70% to 50% on the first try. Headlines now give each scorer's
   changes, e.g. `correct: 1 fixed, 2 broken (80% -> 70%)`.
3. **Replayed runs lost their labels.** A v2 replay was saved without its `prompt_version`
   (and agent version and tags), so v1 and v2 runs could not be told apart.
4. **The example's own help-centre search** only matched the literal words "refund",
   "shipping" and "warranty". "shoe returns" or "express delivery" found nothing, so the
   first v2 eval blamed the prompt for a tool bug. It now matches synonyms, and v1 was
   re-recorded with the same tools.
