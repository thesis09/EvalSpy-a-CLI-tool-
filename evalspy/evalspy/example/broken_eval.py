"""
eval/evaluate.py  —  H100 Direct Inference Edition
────────────────────────────────────────────────────
Evaluates the fine-tuned Forge model DIRECTLY on the H100.
No FastAPI server. No llama.cpp. No GGUF.
Uses the merged HuggingFace model in bfloat16 (full precision).

WHY this is better than the server approach for evaluation:
  • Full bfloat16 precision — no quantization artifacts from GGUF Q4_K_M
  • Native Gemma 3 SentencePiece tokenizer — zero token-dropping bugs
  • No HTTP overhead between ask() and the model
  • H100 80 GB fits Gemma 3 27B bfloat16 (~54 GB) comfortably

Run as a notebook cell:
  exec(open("eval/evaluate.py").read())

Or from terminal:
  python eval/evaluate.py

Metrics:
  1. HumanEval  — pass@1  (50 Python problems)
  2. DebugBench — accuracy (50 samples, 5 target languages)
  3. Spot checks — 5 manual quality prompts
"""

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

# ── Model path ────────────────────────────────────────────────────────────────
# Point at the MERGED safetensors directory, NOT the GGUF file.
# Run merge_and_export.py first if this directory doesn't exist.
MERGED_MODEL_PATH = "output/gemma3-forge-v1/merged"

RESULTS_FILE = Path("eval/results.json")

# ── Generation settings (matches main.py) ────────────────────────────────────
MAX_NEW_TOKENS = 768
TEMPERATURE    = 0.1
TOP_P          = 0.95
TOP_K          = 40
REPEAT_PENALTY = 1.1

# ── System prompt (identical to main.py) ─────────────────────────────────────
SYSTEM_PROMPT = """You are Forge, an elite precision coding assistant with deep expertise in Python, JavaScript, Java, C++, and C.

## ABSOLUTE CODE RULES — never break these:
1. Every function signature MUST include ALL parameter names.
2. Every comparison MUST have both sides.
3. Every variable used in the body MUST be declared in the signature or defined before use.
4. Code blocks MUST be syntactically complete and immediately executable.
5. Never write TODO, FIXME, or "implement this".
6. All code MUST be inside a properly fenced code block with the language tag.
7. Function and variable names MUST be descriptive and follow language conventions.

## RESPONSE STRUCTURE:
1. One-sentence summary.
2. Complete code in a fenced block.
3. Brief explanation (3-6 bullet points).
4. Edge cases — at least 2.

## WHEN WRITING ALGORITHMS:
- State time and space complexity after the code block.
- Show a worked example.

## WHEN DEBUGGING:
- Root cause in one sentence.
- Corrected code with # FIXED: comments.
- Explain WHY the bug occurred.

## LANGUAGE AND FORMATTING:
- Grammatically correct English. Proper punctuation.
- Complexity: O(n log n). Always state both time AND space."""

# ── Global model handles ──────────────────────────────────────────────────────
_model     = None
_tokenizer = None


def load_model():
    """Load merged Gemma 3 27B in bfloat16. H100 80GB fits this with ~26GB to spare."""
    global _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = Path(MERGED_MODEL_PATH)
    if not p.exists():
        print(f"✗ Model not found at {p}")
        print("  Run merge_and_export.py first.")
        sys.exit(1)

    shards   = list(p.glob("*.safetensors"))
    total_gb = sum(s.stat().st_size for s in shards) / 1e9
    print(f"  Path    : {p}")
    print(f"  Size    : {len(shards)} shards, {total_gb:.1f} GB")
    print(f"  Loading … (2–4 min on first run)")

    _tokenizer = AutoTokenizer.from_pretrained(str(p))

    _model = AutoModelForCausalLM.from_pretrained(
        str(p),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    _model.eval()
    print("  ✓ Model loaded\n")


def ask(user_prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Direct inference using native HF transformers.
    apply_chat_template uses Gemma 3's real SentencePiece tokenizer —
    no llama-bpe mapping, no token drops, no variable name corruption.

    FIX: apply_chat_template returns a BatchEncoding dict when return_dict=True.
    Must use **inputs to unpack it for model.generate(), and access
    input_ids via inputs["input_ids"] not inputs.shape.
    """
    import torch

    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]

    # return_dict=True → returns BatchEncoding with input_ids + attention_mask
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,                        # unpacks input_ids + attention_mask
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPEAT_PENALTY,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Slice off input tokens — return only the newly generated part
    input_length = inputs["input_ids"].shape[-1]
    new_tokens   = output_ids[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Code helpers ──────────────────────────────────────────────────────────────
def extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    return text[idx:].strip() if idx != -1 else text.strip()


_CODE_LINE_RE = re.compile(
    r'^(\s*)(def |class |for |while |if |elif |else:|return |import |from |'
    r'try:|except|finally:|with |async |await |yield |'
    r'#include|public |private |static |void |int |float |char |bool |'
    r'function |const |let |var )',
    re.MULTILINE,
)

def _sniff_language(code: str) -> str:
    if re.search(r'^\s*(def |import |from )\w+', code, re.M):      return "python"
    if re.search(r'#include\s*[<"]|std::|cout',   code, re.M):     return "cpp"
    if re.search(r'public\s+class\s+\w+',          code, re.M):    return "java"
    if re.search(r'(function\s+\w+|const\s+\w+=|=>)', code, re.M): return "javascript"
    if re.search(r'#include\s*<stdio|printf\s*\(', code, re.M):    return "c"
    return "python"

def _wrap_bare_code(text: str) -> str:
    """Wrap bare code in fences if the model forgot to add them."""
    if "```" in text:
        return text
    lines = text.splitlines()
    first_code = last_code = None
    for i, line in enumerate(lines):
        is_code = bool(_CODE_LINE_RE.match(line)) or (
            first_code is not None and re.match(r'^\s{4,}\S', line)
        )
        if is_code:
            if first_code is None:
                first_code = i
            last_code = i
    if first_code is None:
        return text
    while last_code + 1 < len(lines) and lines[last_code + 1].strip() == "":
        last_code += 1
    lang = _sniff_language("\n".join(lines[first_code:last_code + 1]))
    return "\n".join(
        lines[:first_code] + [f"```{lang}"]
        + lines[first_code:last_code + 1] + ["```"]
        + lines[last_code + 1:]
    )


# ── 1. HumanEval ─────────────────────────────────────────────────────────────
def _smart_assemble(fn_prompt: str, model_response: str) -> str:
    """
    Correctly build the final executable code from fn_prompt + model response.

    ROOT CAUSE of 0% pass rate:
      Old code did: full_code = fn_prompt + "\n" + extract_python_code(response)
      When the model returns the COMPLETE function (most common at low temperature),
      extract_python_code() returns the full def block. Prepending fn_prompt then
      creates a DUPLICATE function definition → IndentationError or wrong behaviour.

    This function handles all 3 cases:
      A) Model returned complete function with def → use model output directly
      B) Model returned only the body (no def)    → indent and attach to stub
      C) Garbled / empty output                   → stub + pass (safe fallback)
    """
    # Pull code out of fences
    m = re.search(r"```(?:python)?\n(.*?)```", model_response, re.DOTALL)
    code = m.group(1).strip() if m else model_response.strip()

    if not code:
        return fn_prompt.rstrip() + "\n    pass"

    # Case A: model re-declared the function
    fn_name_match = re.search(r"def\s+(\w+)\s*\(", fn_prompt)
    if fn_name_match and f"def {fn_name_match.group(1)}" in code:
        return code   # standalone complete function — use as-is

    # Case B: body only, no def — re-indent to 4 spaces and attach
    if not code.lstrip().startswith("def "):
        indented = "\n".join(
            ("    " + line) if line.strip() else ""
            for line in code.splitlines()
        )
        return fn_prompt.rstrip() + "\n" + indented

    # Fallback
    return fn_prompt.rstrip() + "\n    pass"


def run_python_code(code: str, test: str, timeout: int = 10) -> bool:
    full = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        res = subprocess.run([sys.executable, fname],
                             capture_output=True, timeout=timeout)
        return res.returncode == 0
    except Exception:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)


def _ask_humaneval(fn_prompt: str) -> str:
    """
    Separate ask() for HumanEval — no long system prompt.

    WHY: The full SYSTEM_PROMPT adds ~800 tokens of instructions.
    For short completion tasks (HumanEval stubs are 5-15 lines),
    this confuses the model. A minimal, direct prompt works far better.
    """
    import torch

    # Minimal prompt — just tell it to complete the function
    user_msg = (
        f"Complete this Python function. "
        f"Return ONLY the completed function inside a ```python``` block. "
        f"Do not add any explanation.\n\n"
        f"```python\n{fn_prompt}\n```"
    )
    messages = [{"role": "user", "content": user_msg}]

    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens   = output_ids[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def eval_humaneval(n: int = 50) -> dict:
    print(f"\n[HumanEval] {n} problems ...")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("openai/human-eval", split="test").select(range(n))
        except Exception:
            ds = load_dataset("openai_humaneval", split="test").select(range(n))
    except Exception as e:
        print(f"  Could not load: {e}")
        return {}

    passed  = 0
    results = []

    for row in tqdm(ds, desc="  humaneval"):
        fn_prompt = row["prompt"]
        test_code = row["test"]
        task_id   = row["task_id"]

        response  = _ask_humaneval(fn_prompt)
        full_code = _smart_assemble(fn_prompt, response)
        ok        = run_python_code(full_code, test_code)
        if ok:
            passed += 1
        results.append({"task_id": task_id, "passed": ok})

        # Live progress every 10 problems
        if len(results) % 10 == 0:
            print(f"    {len(results)}/{n}  pass@1 so far: {passed/len(results):.1%}")

    pass_at_1 = passed / n
    print(f"  pass@1 = {pass_at_1:.2%}  ({passed}/{n})")
    return {"pass@1": pass_at_1, "passed": passed, "total": n, "results": results}


# ── 2. DebugBench ─────────────────────────────────────────────────────────────
def _get_buggy_fixed(row: dict) -> tuple[str, str]:
    buggy_keys = ["buggy_code", "bug_code", "code", "input"]
    fixed_keys = ["fixed_code", "correct_code", "target", "output"]
    buggy = next((row[k] for k in buggy_keys if k in row and row[k]), "")
    fixed = next((row[k] for k in fixed_keys if k in row and row[k]), "")
    return buggy, fixed

ALLOWED_LANGS = {"python", "c", "cpp", "c++", "java", "javascript", "js"}

def eval_debug(n: int = 50) -> dict:
    print(f"\n[DebugBench] {n} samples ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Rtian/DebugBench", split="test").select(range(n))
    except Exception as e:
        print(f"  Could not load: {e}")
        return {}

    if len(ds) > 0:
        print(f"  Fields: {list(ds[0].keys())}")

    correct = processed = skipped = 0
    results = []

    for row in tqdm(ds, desc="  debugbench"):
        lang = (row.get("language") or row.get("lang") or "").lower()
        if lang not in ALLOWED_LANGS:
            skipped += 1
            continue
        buggy, fixed = _get_buggy_fixed(row)
        if not buggy or not fixed:
            skipped += 1
            continue

        bug_type  = (row.get("bug_type") or row.get("error_type") or "bug").lower()
        processed += 1
        response  = ask(
            f"Find the bug in this {lang} code and show the fixed version:\n\n"
            f"```{lang}\n{buggy}\n```",
            max_new_tokens=512,
        )
        fixed_tokens = set(fixed.lower().split()[:30])
        resp_tokens  = set(response.lower().split())
        overlap      = len(fixed_tokens & resp_tokens) / max(len(fixed_tokens), 1)
        ok           = overlap > 0.5
        if ok:
            correct += 1
        results.append({"bug_type": bug_type, "overlap": round(overlap, 3), "passed": ok})

    accuracy = correct / processed if processed > 0 else 0.0
    print(f"  accuracy = {accuracy:.2%}  ({correct}/{processed} correct, {skipped} skipped)")
    return {"accuracy": accuracy, "correct": correct,
            "processed": processed, "skipped": skipped}


# ── 3. Spot checks ────────────────────────────────────────────────────────────
SPOT_CHECKS = [
    {
        "name":             "list_flattening",
        "prompt":           "Write a Python function that flattens a nested list of arbitrary depth.",
        "expect_keywords":  ["def", "flatten", "isinstance", "list", "yield"],
    },
    {
        "name":   "sql_injection",
        "prompt": (
            "What is the security bug in this code and how do you fix it?\n\n"
            "```python\n"
            "user_input = input('Enter name: ')\n"
            "query = 'SELECT * FROM users WHERE name = ' + user_input\n"
            "cursor.execute(query)\n"
            "```"
        ),
        "expect_keywords": ["sql injection", "parameterized", "placeholder"],
    },
    {
        "name":   "off_by_one",
        "prompt": (
            "This loop skips the first element. Debug and fix it:\n\n"
            "```python\n"
            "arr = [10, 20, 30]\n"
            "for i in range(1, len(arr)):\n"
            "    print(arr[i])\n"
            "```"
        ),
        "expect_keywords": ["range(0", "range(len", "first", "index"],
    },
    {
        "name":   "async_await",
        "prompt": (
            "Write an async Python function that fetches JSON from 3 URLs "
            "concurrently using aiohttp and asyncio.gather."
        ),
        "expect_keywords": ["async def", "await", "asyncio.gather", "aiohttp"],
    },
    {
        "name":   "time_complexity",
        "prompt": (
            "This function is O(n²). Rewrite it to be O(n):\n\n"
            "```python\n"
            "def has_duplicate(arr):\n"
            "    for i in range(len(arr)):\n"
            "        for j in range(i+1, len(arr)):\n"
            "            if arr[i] == arr[j]:\n"
            "                return True\n"
            "    return False\n"
            "```"
        ),
        "expect_keywords": ["set(", "O(n)", "seen"],
    },
]

def eval_spot_checks() -> dict:
    print(f"\n[Spot Checks] {len(SPOT_CHECKS)} prompts ...")
    results = []
    for check in SPOT_CHECKS:
        response  = ask(check["prompt"], max_new_tokens=768)
        full_text = response.lower()
        keywords  = check["expect_keywords"]
        hits      = sum(1 for kw in keywords if kw.lower() in full_text)
        total_k   = len(keywords)
        score     = hits / total_k
        matched   = [kw for kw in keywords if kw.lower() in full_text]
        results.append({
            "name": check["name"], "score": round(score, 3),
            "hits": hits, "total": total_k, "matched": matched,
            "response_preview": _wrap_bare_code(response)[:400],
        })
        sym = "✓" if score >= 0.6 else "~" if score >= 0.3 else "✗"
        print(f"  {sym} {check['name']:25s}  {score:.0%}  ({hits}/{total_k})  matched={matched}")

    mean = sum(r["score"] for r in results) / len(results)
    print(f"  Mean: {mean:.1%}")
    return {"mean_score": round(mean, 3), "checks": results}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Forge — Evaluation  (H100 · bfloat16 · Direct HF)")
    print("=" * 58)
    print(f"  Model      : {MERGED_MODEL_PATH}")
    print(f"  Temp={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}  rep_pen={REPEAT_PENALTY}")
    print()

    print("[0/4] Loading model ...")
    load_model()

    print("[0/4] Smoke test ...")
    smoke = ask("Reply with only the word: ready", max_new_tokens=10)
    print(f"  Model says: {repr(smoke[:80])}")
    if not smoke.strip():
        print("  ✗ Empty response — check model path and load.")
        sys.exit(1)
    print("  ✓ Ready\n")

    t0 = time.time()
    he = eval_humaneval(n=50)
    db = eval_debug(n=50)
    sp = eval_spot_checks()
    elapsed = time.time() - t0

    summary = {
        "model":              MERGED_MODEL_PATH,
        "precision":          "bfloat16",
        "inference_mode":     "direct_hf",
        "temperature":        TEMPERATURE,
        "humaneval_pass@1":   he.get("pass@1"),
        "debug_accuracy":     db.get("accuracy"),
        "spot_mean_coverage": sp.get("mean_score"),
        "eval_time_minutes":  round(elapsed / 60, 1),
        "full":               {"humaneval": he, "debug": db, "spot": sp},
    }

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*58}")
    print(f"  RESULTS")
    print(f"{'='*58}")
    print(f"  HumanEval pass@1  : {he.get('pass@1', 0):.2%}")
    print(f"  Debug accuracy    : {db.get('accuracy', 0):.2%}")
    print(f"  Spot mean         : {sp.get('mean_score', 0):.1%}")
    print(f"  Time              : {summary['eval_time_minutes']} min")
    print(f"\n  Saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()