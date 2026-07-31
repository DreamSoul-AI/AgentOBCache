# ReAct OBCache Agent Project Progress

Last updated: 2026-07-28

This document records the current state of `react_obcache_agent`. The project has moved from a single-file ReAct demo into an experimental framework for tool-using ReAct agents, context baselines, and an OBCache v0 prompt-level simulation.

The central research question is:

> In multi-step ReAct agents, tool observations make the prompt grow. Can we reduce repeated context or cached tool evidence while preserving answer quality, and later convert that idea into real KV-cache reuse/pruning?

Important status note: the current OBCache v0 is not true low-level KV-cache pruning. It is a prompt/context simulation that prepares the baseline logic for later `past_key_values` work.

## Current Project Structure

```text
react_obcache_agent/
  main.py
  requirements.txt
  PROJECT_PROGRESS.md
  PROJECT_PROGRESS-v0.md

  agent/
    react_agent.py
    prompt_builder.py
    action_parser.py
    trajectory.py

  tools/
    wiki_tool.py
    paper_tool.py
    lc_tools.py

  models/
    base_backend.py
    hf_backend.py
    obcache_backend.py

  memory/
    full_context.py
    window_context.py
    segment_context.py

  langchain_runtime/
    graph.py
    state.py
    model_node.py
    tool_node.py
    tool_call_parser.py
    message_serializer.py
    context_selector.py

  experiments/
    run_batch.py
    run_tool_batch.py

  data/
    questions.jsonl
    tool_tasks.jsonl

  logs/
    batch_summary.json
    tool_batch_summary.json
    tool_batch_results_langgraph_json.jsonl
    tool_result_cache.json
```

`PROJECT_PROGRESS-v0.md` is the older record. This file describes the current logic.

## Runtime Tracks

### Original ReAct Baseline

The earlier modular ReAct baseline lives in:

```text
agent/react_agent.py
agent/prompt_builder.py
agent/action_parser.py
tools/wiki_tool.py
models/hf_backend.py
memory/full_context.py
memory/window_context.py
memory/segment_context.py
experiments/run_batch.py
```

It runs Wikipedia-style Thought, Action, Observation, Answer loops and logs token growth, generation time, and context statistics.

Run it with:

```bash
py react_obcache_agent/experiments/run_batch.py
```

### Current Tool-ReAct Runtime

The newer runtime lives in `langchain_runtime/` and is the main path for OBCache-style experiments.

It uses LangGraph only for orchestration:

```text
agent node -> tool node -> agent node -> ... -> final answer
```

It is intentionally not a black-box LangChain agent. Prompt serialization, parsing, forced tool-call fallback, context selection, logging, and evaluation are controlled by local code.

Main runner:

```bash
py react_obcache_agent/experiments/run_tool_batch.py
```

## Current ReAct Flow

For each task in `data/tool_tasks.jsonl`:

1. `run_tool_batch.py` loads the question and expected answer.
2. `graph.py` starts the LangGraph loop.
3. `model_node.py` selects the context and serializes the model prompt.
4. `hf_backend.py` generates raw model output.
5. `tool_call_parser.py` parses JSON tool calls or final answers.
6. `tool_node.py` executes tools and appends tool results.
7. The graph loops until a final answer or max tool-step limit.
8. Per-step logs and per-task results are written to `logs/`.

State fields are defined in `state.py`:

```text
messages
tool_logs
cache_logs
step_count
tool_replay_mode
tool_result_cache
method_name
context_mode
keep_last_tool_steps
```

## Tools

Current LangGraph tools are exposed through `tools/lc_tools.py`:

```text
wiki_search
wiki_lookup
paper_search
calculator
```

`wiki_search` and `wiki_lookup` use Wikipedia.

`paper_search` handles paper/method/authorship queries such as ReAct prompting method questions.

`calculator` handles arithmetic questions.

## Tool Replay Cache

Tool outputs are cached to reduce network instability and make experiments repeatable.

Cache file:

```text
logs/tool_result_cache.json
```

Modes:

```text
TOOL_REPLAY_MODE=live    # use live tools, do not write cache
TOOL_REPLAY_MODE=record  # use live tools and write cache
TOOL_REPLAY_MODE=replay  # read tool results from cache
```

Default mode is `replay`.

## Compared Methods

`experiments/run_tool_batch.py` currently compares:

```text
full_tool_react
window_tool_react_1
window_tool_react_2
window_tool_react_3
segment_tool_react
obcache_tool_react_v0
```

### full_tool_react

Keeps all conversation messages and all tool results. This is the main quality baseline.

### window_tool_react_1 / 2 / 3

Keeps the user/system prefix and the most recent N tool steps. This is the simple window baseline.

Observation: `window_tool_react_1` prunes most aggressively but can drop important multi-hop evidence.

### segment_tool_react

Keeps the overall message structure but removes older tool result contents according to a segment policy. This tests whether old observations can be omitted while preserving enough recent evidence.

### obcache_tool_react_v0

This is the current OBCache simulation baseline. It does not prune `past_key_values` yet.

Current policy:

```text
keep the first tool result as anchor evidence
keep the most recent N tool results
replace older middle tool results with a placeholder
```

Placeholder:

```text
[cached tool result omitted from prompt]
```

The first tool result is kept because multi-hop tasks often put subject identity and key answer evidence in the first search result.

## Logging

Each model step logs into `cache_logs`:

```text
method_name
context_mode
prompt_tokens
full_prompt_tokens
selected_prompt_tokens
context_pruned_tokens
context_pruned_ratio
output_tokens
generation_time
gpu_memory_allocated_mb
gpu_memory_reserved_mb
full_message_count
selected_message_count
total_tool_results
kept_tool_results
cached_tool_results
omitted_tool_results
parse_error
forced_tool_call
answer_source
raw_output
```

Each tool call logs into `tool_logs`:

```text
tool_name
arguments
success
latency_seconds
result_preview
was_cached
replay_mode
cache_key
```

## Evaluation Files

Main outputs:

```text
logs/tool_batch_summary.json
logs/tool_batch_results_langgraph_json.jsonl
```

`tool_batch_summary.json` stores method-level metrics.

`tool_batch_results_langgraph_json.jsonl` stores per-question predictions, tool calls, tool logs, cache logs, and trajectories.

Important metrics:

```text
exact_match_accuracy
contains_match_accuracy
avg_token_f1
runtime_finish_rate
tool_execution_success_rate
tool_selection_exact_match_rate
raw_parse_success_rate
forced_tool_call_rate
replay_cache_hit_rate
avg_prompt_tokens
total_context_pruned_ratio
avg_generation_time_per_question
avg_end_to_end_time_per_question
```

For this project, `contains_match_accuracy` and `avg_token_f1` are usually more informative than exact match, because many correct predictions are full sentences while gold answers are short entities.

## Latest Completed 30-Task Result

Latest completed full run before the OBCache v0 anchor-evidence fix:

```text
full_tool_react:       exact=0.5667, contains=1.0000, prompt=741.16, pruned=0.0000, gen/q=2.32s
window_tool_react_1:   exact=0.5333, contains=0.9333, prompt=657.90, pruned=0.1526, gen/q=2.66s
window_tool_react_2:   exact=0.5667, contains=1.0000, prompt=736.49, pruned=0.0063, gen/q=2.33s
window_tool_react_3:   exact=0.5667, contains=1.0000, prompt=741.16, pruned=0.0000, gen/q=2.31s
segment_tool_react:    exact=0.6000, contains=0.9667, prompt=667.83, pruned=0.0989, gen/q=2.31s
obcache_tool_react_v0: exact=0.6000, contains=0.9667, prompt=671.10, pruned=0.0945, gen/q=2.33s
```

Interpretation:

```text
Full keeps all evidence and reaches contains=1.0.
Window-1 prunes most but loses quality.
Window-2 and Window-3 preserve quality but prune almost nothing on this task set.
Segment and OBCache v0 save about 9-10% prompt context with small quality loss.
Speed is similar because OBCache v0 is still prompt simulation, not real KV reuse.
```

## Error Analysis

The main OBCache v0 drop was `tool_018`.

Question:

```text
Search Alan Turing, then use wiki_lookup for artificial intelligence, and answer one field he is widely considered the father of.
```

Gold:

```text
theoretical computer science
```

Full answer:

```text
Alan Turing is widely considered the father of theoretical computer science.
```

Old OBCache v0 answer:

```text
Alan Turing is widely considered the father of artificial intelligence.
```

Cause:

```text
Both methods used wiki_search and wiki_lookup.
The old OBCache v0 omitted the first wiki_search result.
The latest lookup result focused on artificial intelligence.
The model was pulled toward the latest evidence and answered the wrong field.
```

Fix applied in `langchain_runtime/context_selector.py`:

```text
obcache_v0 now keeps the first tool result as anchor evidence.
obcache_v0 also keeps the latest N tool results.
Only middle older tool results are replaced by placeholders.
```

After this fix, rerun the full 30-task benchmark to update `tool_batch_summary.json`.

## Useful Commands

Run all methods on all tool tasks:

```bash
cd ~/Desktop/kvcache
unset QUESTION_LIMIT
unset TOOL_METHODS
unset TASK_IDS
export TOOL_REPLAY_MODE=replay
py react_obcache_agent/experiments/run_tool_batch.py
```

Run a small smoke test:

```bash
export QUESTION_LIMIT=3
export TOOL_REPLAY_MODE=replay
py react_obcache_agent/experiments/run_tool_batch.py
```

Compare only Full and OBCache v0:

```bash
export TOOL_METHODS=full_tool_react,obcache_tool_react_v0
export TOOL_REPLAY_MODE=replay
py react_obcache_agent/experiments/run_tool_batch.py
```

Run one task only:

```bash
export TASK_IDS=tool_018
export TOOL_METHODS=full_tool_react,obcache_tool_react_v0
export TOOL_REPLAY_MODE=replay
py react_obcache_agent/experiments/run_tool_batch.py
```

Clear filters before a full run:

```bash
unset QUESTION_LIMIT
unset TOOL_METHODS
unset TASK_IDS
```

## Current Limitations

```text
OBCache v0 is prompt-level simulation only.
No true past_key_values reuse yet.
No low-level KV pruning yet.
No prefill/decode time split yet.
Average prompt length is still short, around 700-800 tokens.
The current tool dataset has only 30 tasks.
Some outputs still require forced tool calls or fallback parsing.
Exact match is harsh for correct full-sentence answers.
```

## Next Steps

### Step 1: Rerun after OBCache v0 anchor fix

Compare Full and OBCache v0 again:

```text
contains_match_accuracy
avg_token_f1
avg_prompt_tokens
total_context_pruned_ratio
avg_generation_time_per_question
```

Expected outcome:

```text
OBCache v0 contains accuracy should improve.
Prompt pruning may decrease slightly because anchor evidence is retained.
```

### Step 2: Expand evaluation tasks

The current 30-task set is enough for debugging but too small for conclusions.

Recommended expansion:

```text
50-100 tool tasks
more 3-5 hop questions
longer Wikipedia observations
more first-evidence vs latest-evidence conflict cases
more paper_search + wiki_search mixed tasks
```

### Step 3: Add true KV reuse baseline

Before real OBCache pruning, implement no-prune KV reuse:

```text
hf_full_recompute
kv_reuse_no_prune
obcache_v0_prompt_simulation
```

Metrics to add:

```text
prefill_time
decode_time
kv_reuse_hit_tokens
kv_reuse_miss_tokens
kv_cache_layers
kv_cache_memory_mb
output_parity_with_full_recompute
```

### Step 4: Segment-aware KV pruning

After `kv_reuse_no_prune` works, move from prompt simulation to real OBCache pruning:

```text
segment messages and tool results
assign keep/cache/omit policy per segment
preserve immutable prompt and current question
preserve anchor evidence and latest evidence
reuse KV for stable prefix tokens
prune or omit old middle evidence tokens
compare quality and speed against full/window/segment baselines
```

## Current Status

```text
Baseline ReAct: implemented
Tool ReAct runtime: implemented
Wikipedia/paper/calculator tools: implemented
Replay cache: implemented
Batch evaluation: implemented
Full/window/segment baselines: implemented
OBCache v0 prompt simulation: implemented
OBCache v0 anchor-evidence fix: implemented
TASK_IDS single-task debug filter: implemented
True KV reuse: not implemented yet
True KV pruning: not implemented yet
```

The project is ready for a full 30-task rerun after the OBCache v0 anchor-evidence fix, then the next engineering milestone is a true no-prune KV reuse baseline.