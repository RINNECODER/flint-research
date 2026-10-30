# Flint SFT data v2 (agentic coding from frontier-model traces)

Files: `train.jsonl` (2053 ex, 5,832,638 tokens, 2,934,025 trained), `val.jsonl` (150 ex, 448,193 tokens), `maskcheck.log` (decoded mask check), `SOURCES.md` (candidate table), `build/` (scripts; `fetch.sh` re-downloads the raw inputs, which were deleted).
Each line `{"src","ids","mask"}`, same as `../data`. Rendering, template, `<think>`, EOS 150001 appended after every assistant turn (incl. tool-call turns), no BOS, mask=1 on assistant output only: identical to `../data/DATA.md` (same `render.py`, same tokenizer/template snapshot c0d2388f of dima6312/Flint-40B-A2.5B-abliterated-Rec-4bit).

## Sources (all public HF, licences checked, none forbids training use)
| src prefix | dataset | licence | teacher (as claimed) |
|---|---|---|---|
| fable5-dsf-* | DSFFGFG456/fable-5-coding-and-debugging-traces | cc-by-4.0 | claude-fable-5 (OpenRouter, unattested) |
| fable5-cc-armand0e-* | armand0e/claude-fable-5-claude-code | mit | claude-fable-5 (per-message model field checked) |
| fable5-cc-alin-* | AlinCiocan/fable-5-claude-code-traces | cc-by-4.0 | claude-fable-5 (checked) |
| fable5-cursor-* | TeichAI/Fable-5-Cursor-Traces | apache-2.0 | claude-fable-5 (card) |
| opus46-47-chat-coding | angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k, code_train.jsonl category=coding | apache-2.0 | claude-opus-4-6/4-7; `<think>` is synthetic by the card's own admission |
| (decontam only) | evalplus/humanevalplus, evalplus/mbppplus | apache-2.0 | |
`-full` = whole trajectory fits in 6,144 tokens; `-prefix` = trajectory cut at an assistant-turn boundary (see below). Rejected sources: SOURCES.md.

## Conversion
- **One toolset** for every agentic source (`bash(command,timeout)`, `read(path,offset,limit)`, `write(path,content)`, `edit(path,edits[{oldText,newText}])`, `grep(pattern,path)`, `glob(pattern,path)`, `web_search(query)`, `web_fetch(url,prompt)`), announced through the Flint tools block. Mapping: Claude Code `Bash/Read/Write/Edit/Grep/Glob/WebSearch/WebFetch`, pi lowercase tools, Cursor `run_terminal_cmd/read_file/write_file(new_string)/grep_search/glob/read_url/web_search` -> these. Claude `Edit` becomes a one-element `edits` list (`replace_all=true` -> cut). `description` args dropped. Cursor `read_file` with an end_line > 1500 lines becomes "read to end". Cursor `todo_write` calls are removed together with their results.
- Tool calls render as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` through the stock template; parallel calls keep their order and results are re-ordered to match. Non-string args (the `edits` array) are JSON.
- System prompt replaced by one short fixed prompt (the originals were 10-50k chars of harness text). Chat examples keep their own tutor-style system prompt and have no tools block.
- **Unmappable tools** (ToolSearch, Task*, Skill, Agent/`task`, Monitor, PowerShell, MCP preview, fictional-API tools, ...): the trajectory is cut right before the first such call (never dropped mid-turn); examples needing none of bash/write/edit are dropped.
- Claude Code raw sessions: sidechains skipped, `system-reminder` blocks stripped, split into one segment per real user prompt; assistant messages merged by message id; any non-`claude-fable-5` assistant message cuts the segment; vague restarts ("continue", <15 chars) and task-notification prompts dropped.
- Length rule: **max 6,144 tokens** (whole rendered example incl. tools block and EOS). Longer trajectories are not truncated mid-turn; the longest prefix that ends on an assistant turn within the limit is kept (the last turn may be a tool call with no result after it, as in the old set). 1 assistant turn minimum for DSF/chat, 2 for CC/Cursor.
- Decontamination: word-level lowercase 13-gram overlap against HumanEval+ and MBPP+ (prompt + canonical solution) over all text incl. tool calls and results: 2 dropped (chat), 0 elsewhere. Exact-ish dup filter (first user message + final assistant text): 20 dropped.
- Secret scan of the decoded train set (api-key / AWS / GitHub token / private-key / email patterns): only placeholder emails (alice@example.com style).

## Counts and token stats
Train:
| src | n | tokens | mean | median | max | trained tokens |
|---|---|---|---|---|---|---|
| fable5-cc-alin-full | 23 | 58,919 | 2562 | 2284 | 4964 | 27,298 |
| fable5-cc-alin-prefix | 8 | 38,856 | 4857 | 4962 | 5895 | 10,292 |
| fable5-cc-armand0e-full | 59 | 160,370 | 2718 | 2095 | 6120 | 78,933 |
| fable5-cc-armand0e-prefix | 72 | 287,145 | 3988 | 4044 | 6091 | 106,493 |
| fable5-cursor-full | 16 | 68,898 | 4306 | 5037 | 6098 | 42,612 |
| fable5-cursor-prefix | 161 | 826,758 | 5135 | 5783 | 6140 | 421,847 |
| fable5-dsf-full | 438 | 1,760,909 | 4020 | 4135 | 6143 | 490,001 |
| fable5-dsf-prefix | 243 | 885,190 | 3643 | 3610 | 6141 | 132,987 |
| opus46-47-chat-coding | 1033 | 1,745,593 | 1690 | 1265 | 5888 | 1,623,562 |
Total train: n=2053, tokens=5,832,638, trained=2,934,025 (50%). Agentic (tool-call) examples 1,020 = 50%, 70% of tokens; chat-coding 1,033.
Val:
| src | n | tokens | mean | median | max | trained tokens |
|---|---|---|---|---|---|---|
| fable5-cc-alin-full | 1 | 5,676 | 5676 | 5676 | 5676 | 2,296 |
| fable5-cc-alin-prefix | 1 | 951 | 951 | 951 | 951 | 295 |
| fable5-cc-armand0e-full | 3 | 7,927 | 2642 | 2806 | 4299 | 4,470 |
| fable5-cc-armand0e-prefix | 7 | 32,517 | 4645 | 5284 | 6060 | 11,163 |
| fable5-cursor-prefix | 12 | 61,004 | 5084 | 5916 | 6142 | 35,097 |
| fable5-dsf-full | 36 | 144,631 | 4018 | 4604 | 6125 | 39,858 |
| fable5-dsf-prefix | 14 | 48,603 | 3472 | 3191 | 6083 | 6,458 |
| opus46-47-chat-coding | 76 | 146,884 | 1933 | 1501 | 5941 | 137,027 |
Total val: n=150, tokens=448,193. Split is random, proportional per source (not per trajectory-disjoint across the Claude Code segments of one session; segments of one session can appear on both sides).

## Mask check (decoded; full log in maskcheck.log)
Every example ends in EOS (150001) with mask=1 and every contiguous mask=1 run ends in EOS (checked on all 2053 train rows). Two examples, first 3 trained runs, abridged:
```
==================== opus46-47-chat-coding len 557 trained 525 assistant runs 1
--- trained run 0 [32:557] ---
<think>The user is asking about __new__ vs __init__ in Python, which is a subtle topic that most Python developers don't fully understand. The key distinction is that __new__ creates the instance (it's a constructor) while __init__ initializes it (it's an initializer). For everyday code, only __init__ matters — __new__ is the default inherited from object and returns a new instance automatically. But __new__ becomes essential in specific patterns. The Singleton pattern is the most intuitive exam
--- untrained head (first 60 tokens) ---
<|system|>You are a concise coding assistant. Provide working code with brief explanations.<|user|>What is the difference between `__new__` and `__init__` in Python?<|assistant|><think>The user is asking about __new__ vs __init__ in Python, which is a subtle topic that most Python developers don't fully understand
--- context just before run 1 (mask=0) ---

==================== fable5-dsf-full len 5562 trained 1572 assistant runs 4
--- trained run 0 [1311:1358] ---
I'll read the test file first to pin down the exact contract, then check for existing files.<tool_call>read<arg_key>path</arg_key><arg_value>test_hooks_runtime.ts</arg_value></tool_call><tool_call>bash<arg_key>command</arg_key><arg_value>ls . && node --version</arg_value></tool_call><|end_of_text|>
--- trained run 1 [3866:4978] ---
<tool_call>write<arg_key>path</arg_key><arg_value>hooks_runtime.ts</arg_value><arg_key>content</arg_key><arg_value>// hooks_runtime.ts — a miniature React-style hooks runtime.\n// Plain TypeScript under node's type stripping: annotations only.\n\ntype HookName = "useState" | "useReducer" | "useMemo";\n\ntype Slot = {\n  type: HookName;\n  state?: unknown;\n  queue?: unknown[];\n  setter?: unknown;\n  reducer?: (state: unknown, action: unknown) => unknown;\n  deps?: unknown[];\n  value?: unknown;\n};\n\ntype Ro
--- trained run 2 [4987:5012] ---
<tool_call>bash<arg_key>command</arg_key><arg_value>node --test test_hooks_runtime.ts 2>&1 | tail -15</arg_value></tool_call><|end_of_text|>
--- untrained head (first 60 tokens) ---
<|system|>You are an autonomous coding agent working in the user's repository. Use your tools to read, search, edit and run code. Inspect before you change anything, make focused edits, run the tests or the program to verify, and give a short final summary.<|system|>\n# Tools\n\nYou may call
--- context just before run 1 (mask=0) ---
 });\n  assert.throws(() => self.render({}), { message: "render is not reen
```

## Caveats
- **2053 train examples, below the 3-6k target.** The supply of clean, distinct, license-OK frontier agentic traces is small: the Fable 5 sets are mirrors of ~80 sessions, and Claude Code / Cursor sessions are 50-200k tokens, so under the 6,144 cap they contribute only their opening (prefix) segments. Levers if more is needed (not used, see SOURCES.md): the AGPL Fable-5-traces mirrors (~+100-150), the 513 conceptual coding answers in angrygiraffe (no code block; would make chat 60%), or raising the length cap.
- **Reasoning is rare in the agentic data**: only 178 of 1,020 agentic examples contain a `<think>` (DSF and Cursor `reasoning_content`, short; Claude Code logs have thinking redacted so CC examples are visible-text-only). Chat examples always have `<think>` (~250 tokens, synthetic, Opus-written). A model trained on this will often start tool turns without `<think>`; the generation prompt is bare `<|assistant|>`, so this is not forced either way.
- **Windowed examples**: 484 of 1,020 agentic examples are prefixes. They show the start of a task (explore, first edits), not the verified end state; DSF "verified by tests" holds for the whole trajectory, not for a prefix. Cursor examples are the opening of very long sessions.
- DSF teacher is "unattested" (card: `model_attested False`, `teacher_model` empty, runtime pi via OpenRouter). The text and tool-id style are consistent with Claude, but it was not independently proven.
- 243 DSF pi-format trajectories lack tool results entirely (dataset defect) and were dropped; 1,374 fictional-API tool-calling trajectories were dropped as non-coding.
- Mostly English; code languages follow the sources (Python/TS/Go/Java/Rust/shell heavy). Windows/PowerShell sessions were cut (PowerShell tool unmapped).
- Claude Code/Cursor sessions contain real (anonymized) user prompts with typos, pasted terminal output and long-horizon context ("continue the experiment"); a few are noisy.
- Tool-result text is the real harness output (cat -n style line numbers in Read results, exit-code lines), not rewritten into our tool format; the model sees slightly different result styles per source.
- `stats.json`, `assemble.log`, `conv.log` in build/ hold the raw numbers.
