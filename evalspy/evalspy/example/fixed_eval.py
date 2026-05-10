"""
eval/evaluate_h100_v4.py  —  Forge Full Evaluation Suite
──────────────────────────────────────────────────────────
H100 · bfloat16 · Direct HF inference (no GGUF, no server)

Changes from v3:
  1. MBPP FIX — extract expected function name from assert statements
               and inject it into the prompt. v3 got 9% because the model
               picked its own function name (e.g. minimum_cost_path) while
               the test hardcoded a different name (e.g. min_cost) → NameError.
               Real capability was hidden behind a naming mismatch, not logic failure.
  2. MBPP FIX — also inject any test_imports (e.g. "from math import sqrt")
               so imported helpers are available at execution time.
  3. evalplus FIX — try both known import paths (evalplus changed its internal
               module layout between versions). Falls back gracefully if neither works.
  4. RESULTS_FILE updated to results_v4.json to avoid overwriting v3 output.

Install once before running:
  pip install evalplus
  pip install datasets transformers trl peft bitsandbytes accelerate tqdm

Run:
  python eval/evaluate_h100_v4.py
  # or from a notebook cell:
  exec(open("eval/evaluate_h100_v4.py").read())
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
MERGED_MODEL_PATH = "output/gemma3-forge-v1/merged"
RESULTS_FILE      = Path("eval/results_v4.json")

# ── Generation settings ───────────────────────────────────────────────────────
MAX_NEW_TOKENS = 768
TEMPERATURE    = 0.1
TOP_P          = 0.95
TOP_K          = 40
REPEAT_PENALTY = 1.1

# ── System prompt (Forge identity — used for DebugBench + Spot checks) ────────
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


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load merged Gemma 3 27B in bfloat16. H100 80GB fits this with ~26GB spare."""
    global _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = Path(MERGED_MODEL_PATH)
    if not p.exists():
        print(f"  ✗ Model not found at {p}")
        print("    Run merge_and_export.py first.")
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


# ══════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def ask(user_prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Full Forge inference with system prompt. Used for DebugBench + Spot checks."""
    import torch

    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]

    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPEAT_PENALTY,
            pad_token_id=_tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens   = output_ids[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _ask_code(prompt_text: str, max_new_tokens: int = 512) -> str:
    """
    Minimal prompt — NO system prompt.
    Used for HumanEval, HumanEval+, and MBPP (pure code completion).
    The full SYSTEM_PROMPT adds ~800 tokens and hurts short completion tasks.
    """
    import torch

    messages = [{"role": "user", "content": prompt_text}]

    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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


def _ask_humaneval(fn_prompt: str) -> str:
    """HumanEval / HumanEval+ wrapper. fn_prompt already contains the def stub."""
    user_msg = (
        "Complete this Python function. "
        "Return ONLY the completed function inside a ```python``` block. "
        "Do not add any explanation.\n\n"
        f"```python\n{fn_prompt}\n```"
    )
    return _ask_code(user_msg)


def _ask_mbpp(description: str, func_name: str) -> str:
    """
    MBPP wrapper.

    FIX: Always pass func_name extracted from the test assertions.
    MBPP tests hardcode a specific function name, e.g.:
        assert min_cost([[1,2],[3,4]], 1, 1) == 4
    If the model picks a different name the test raises NameError even if
    the logic is perfect. Telling the model the exact name required fixes this.

    v3 got 9% on MBPP purely because of this naming mismatch.
    Expected real score after this fix: 60-80%.
    """
    user_msg = (
        f"Write a Python function named `{func_name}` that solves this task:\n\n"
        f"{description}\n\n"
        f"The function MUST be named exactly `{func_name}`.\n"
        f"Return ONLY the function inside a ```python``` block. "
        f"Do not add any explanation or example calls."
    )
    return _ask_code(user_msg)


# ══════════════════════════════════════════════════════════════════════════════
#  Code helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    return text[idx:].strip() if idx != -1 else text.strip()


def extract_func_name_from_tests(test_list: list) -> str:
    """
    Extract the expected function name from MBPP assert statements.

    MBPP tests look like:
        assert func_name(arg1, arg2) == expected_value
        assert func_name(arg1) == expected_value

    We grab the name from the first assert that matches.
    Falls back to "solution" if no match (safe default that won't collide).
    """
    for test in test_list:
        # Match: assert some_func_name(
        m = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', test)
        if m:
            return m.group(1)
    return "solution"


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


def _smart_assemble(fn_prompt: str, model_response: str) -> str:
    """
    Build final executable code from fn_prompt + model response.

    Case A: model re-declared the full function → use directly
    Case B: model returned body only (no def)   → indent + attach to stub
    Case C: empty / garbled                     → stub + pass (safe fallback)
    """
    m = re.search(r"```(?:python)?\n(.*?)```", model_response, re.DOTALL)
    code = m.group(1).strip() if m else model_response.strip()

    if not code:
        return fn_prompt.rstrip() + "\n    pass"

    fn_name_match = re.search(r"def\s+(\w+)\s*\(", fn_prompt)
    if fn_name_match and f"def {fn_name_match.group(1)}" in code:
        return code  # Case A

    if not code.lstrip().startswith("def "):
        indented = "\n".join(
            ("    " + line) if line.strip() else ""
            for line in code.splitlines()
        )
        return fn_prompt.rstrip() + "\n" + indented  # Case B

    return fn_prompt.rstrip() + "\n    pass"  # Case C


def run_python_code(code: str, test: str, timeout: int = 10) -> bool:
    """Execute code + test in a subprocess. Returns True if exit code is 0."""
    full = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        res = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return res.returncode == 0
    except Exception:
        return False
    finally:
        Path(fname).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. HumanEval — full 164 problems
# ══════════════════════════════════════════════════════════════════════════════

def eval_humaneval(n: int = 164) -> dict:
    """Standard HumanEval pass@1 — full 164 problem set."""
    print(f"\n[HumanEval] {n} problems (full set) ...")
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

        if len(results) % 10 == 0:
            print(f"    {len(results)}/{n}  pass@1 so far: {passed/len(results):.1%}")

    pass_at_1 = passed / n
    print(f"  pass@1 = {pass_at_1:.2%}  ({passed}/{n})")
    return {"pass@1": pass_at_1, "passed": passed, "total": n, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
#  2. HumanEval+ — same 164 problems, ~80x more test cases each
# ══════════════════════════════════════════════════════════════════════════════

def _load_evalplus():
    """
    FIX: try both evalplus import layouts (changed between versions).
    Layout A (older):  evalplus.data / evalplus.evaluate
    Layout B (newer):  evalplus.data  / evalplus.eval (no 'evaluate' submodule)
    Returns (get_human_eval_plus, evaluate_functional_correctness) or (None, None).
    """
    # Layout A — most common, what docs show
    try:
        from evalplus.data     import get_human_eval_plus
        from evalplus.evaluate import evaluate_functional_correctness
        return get_human_eval_plus, evaluate_functional_correctness
    except ImportError:
        pass

    # Layout B — newer versions restructured the package
    try:
        from evalplus.data import get_human_eval_plus
        from evalplus.eval import evaluate_functional_correctness
        return get_human_eval_plus, evaluate_functional_correctness
    except ImportError:
        pass

    # Layout C — some builds expose a CLI-only interface; use subprocess fallback
    try:
        from evalplus.data import get_human_eval_plus
        # evaluate_functional_correctness not importable — signal caller to use CLI
        return get_human_eval_plus, None
    except ImportError:
        pass

    return None, None


def eval_humaneval_plus(n: int = 164) -> dict:
    """
    HumanEval+ via evalplus.
    Each of the 164 problems has ~80x more test cases than standard HumanEval.
    Models typically score 5-15% lower here. Gap >15% = pattern matching concern.

    Requires: pip install evalplus
    """
    print(f"\n[HumanEval+] {n} problems (evalplus — extended test cases) ...")

    get_hep, eval_fc = _load_evalplus()

    if get_hep is None:
        print("  ✗ evalplus not importable even though it may be installed.")
        print("    Try: pip install --upgrade evalplus")
        print("    Skipping HumanEval+.")
        return {}

    if eval_fc is None:
        # evalplus installed but evaluate_functional_correctness not importable
        # Fall back to CLI: evalplus.evaluate --dataset humaneval --samples <file>
        print("  ⚠ evalplus found but evaluate_functional_correctness not importable.")
        print("    Will generate samples then call evalplus CLI for scoring.")

    problems     = get_hep()
    problem_list = list(problems.items())[:n]

    samples = []
    for task_id, problem in tqdm(problem_list, desc="  humaneval+"):
        fn_prompt = problem["prompt"]
        response  = _ask_humaneval(fn_prompt)
        full_code = _smart_assemble(fn_prompt, response)
        samples.append({"task_id": task_id, "solution": full_code})

    with tempfile.TemporaryDirectory() as tmp:
        import os
        samples_path = os.path.join(tmp, "samples.jsonl")
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        if eval_fc is not None:
            # Direct Python API path
            results = eval_fc(
                sample_file=samples_path,
                k=[1],
                problems=problems,
                base_only=False,
            )
        else:
            # CLI fallback
            cli_result = subprocess.run(
                [sys.executable, "-m", "evalplus.evaluate",
                 "--dataset", "humaneval",
                 "--samples", samples_path],
                capture_output=True, text=True
            )
            print(cli_result.stdout)
            if cli_result.returncode != 0:
                print(f"  ✗ evalplus CLI error:\n{cli_result.stderr}")
                return {}
            # Parse pass@1 from CLI output
            base_match = re.search(r'pass@1.*?:\s*([\d.]+)', cli_result.stdout)
            plus_match = re.search(r'plus.*?pass@1.*?:\s*([\d.]+)', cli_result.stdout, re.I)
            base_score = float(base_match.group(1)) if base_match else 0.0
            plus_score = float(plus_match.group(1)) if plus_match else 0.0
            gap = base_score - plus_score
            print(f"  HumanEval  base  pass@1 : {base_score:.2%}")
            print(f"  HumanEval+ plus  pass@1 : {plus_score:.2%}")
            print(f"  Gap                     : {gap:.2%}")
            return {
                "base_pass@1": round(base_score, 4),
                "plus_pass@1": round(plus_score, 4),
                "gap":         round(gap, 4),
                "n":           n,
            }

    base_score = results.get("pass@1",      0.0)
    plus_score = results.get("plus_pass@1", 0.0)
    gap        = base_score - plus_score

    print(f"  HumanEval  base  pass@1 : {base_score:.2%}")
    print(f"  HumanEval+ plus  pass@1 : {plus_score:.2%}")
    print(f"  Gap (base - plus)       : {gap:.2%}  ", end="")
    if   gap <= 0.05: print("(excellent — tests are actually passing correctly)")
    elif gap <= 0.12: print("(normal — model handles most edge cases)")
    else:             print("(high — model may be passing by pattern, not logic)")

    return {
        "base_pass@1": round(base_score, 4),
        "plus_pass@1": round(plus_score, 4),
        "gap":         round(gap, 4),
        "n":           n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. MBPP — generalization check
# ══════════════════════════════════════════════════════════════════════════════

def eval_mbpp(n: int = 100) -> dict:
    """
    MBPP (Mostly Basic Python Programming) — 374 problems from Google.

    ROOT CAUSE OF v3's 9% score (now fixed):
      MBPP test assertions hardcode a specific function name, e.g.:
          assert min_cost([[1,2],[3,4]], 1, 1) == 4
      v3 told the model "write a Python function" with no name hint.
      The model chose its own name (e.g. minimum_cost_path) → NameError on every test.
      The logic was correct; the name was wrong. Pure eval bug, not model failure.

    THE FIX:
      extract_func_name_from_tests() reads the first assert and pulls the name.
      _ask_mbpp() then tells the model: "The function MUST be named exactly `min_cost`."
      Expected score after fix: 60-80%, which is the true generalization signal.

    WHY MBPP MATTERS:
      If MBPP score is within 10% of HumanEval → model genuinely learned to code.
      If MBPP is 20%+ lower → HumanEval scores were boosted by training data overlap.
    """
    print(f"\n[MBPP] {n} problems (generalization check, function-name fix applied) ...")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset(
                "google-research-datasets/mbpp", "sanitized", split="test"
            ).select(range(n))
        except Exception:
            ds = load_dataset("mbpp", split="test").select(range(n))
    except Exception as e:
        print(f"  Could not load: {e}")
        return {}

    passed  = 0
    results = []

    for row in tqdm(ds, desc="  mbpp"):
        description  = row.get("text", row.get("prompt", ""))
        test_list    = row.get("test_list", row.get("tests", []))
        test_imports = row.get("test_imports", [])   # e.g. ["import math"]
        task_id      = row.get("task_id", len(results))

        if not description or not test_list:
            results.append({"task_id": task_id, "passed": False, "skipped": True})
            continue

        # Extract the exact function name the test expects
        func_name = extract_func_name_from_tests(test_list)

        # Build test code: imports + assert statements
        imports_block = "\n".join(test_imports) + "\n" if test_imports else ""
        test_code     = imports_block + "\n".join(test_list)

        # Ask the model, naming the function explicitly
        response  = _ask_mbpp(description, func_name)
        code      = extract_python_code(response)
        ok        = run_python_code(code, test_code)

        if ok:
            passed += 1
        results.append({
            "task_id":   task_id,
            "func_name": func_name,
            "passed":    ok,
        })

        if len(results) % 10 == 0:
            print(f"    {len(results)}/{n}  pass@1 so far: {passed/len(results):.1%}")

    pass_at_1 = passed / n
    print(f"  pass@1 = {pass_at_1:.2%}  ({passed}/{n})")
    return {"pass@1": pass_at_1, "passed": passed, "total": n, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
#  4. DebugBench — multi-language debugging accuracy
# ══════════════════════════════════════════════════════════════════════════════

def _get_buggy_fixed(row: dict) -> tuple:
    """
    "solution" is listed first — that's the actual field in Rtian/DebugBench.
    Without it first, every row returns fixed="" and gets skipped.
    """
    buggy_keys = ["buggy_code", "bug_code", "code", "input"]
    fixed_keys = ["solution", "fixed_code", "correct_code", "target", "output"]
    buggy = next((row[k] for k in buggy_keys if k in row and row[k]), "")
    fixed = next((row[k] for k in fixed_keys if k in row and row[k]), "")
    return buggy, fixed


ALLOWED_LANGS = {"python", "c", "cpp", "c++", "java", "javascript", "js"}


def eval_debug(n: int = 50) -> dict:
    """
    DebugBench accuracy via token-overlap proxy.
    NOTE: this is a directional metric, not execution-based.
    A proper implementation would run the fixed code against test cases.
    """
    print(f"\n[DebugBench] {n} samples ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Rtian/DebugBench", split="test").select(range(n))
    except Exception as e:
        print(f"  Could not load: {e}")
        return {}

    if len(ds) > 0:
        print(f"  Fields : {list(ds[0].keys())}")

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

        response     = ask(
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
        results.append({
            "bug_type": bug_type,
            "overlap":  round(overlap, 3),
            "passed":   ok,
        })

    accuracy = correct / processed if processed > 0 else 0.0
    print(f"  accuracy = {accuracy:.2%}  ({correct}/{processed} correct, {skipped} skipped)")
    return {
        "accuracy":  accuracy,
        "correct":   correct,
        "processed": processed,
        "skipped":   skipped,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  5. Spot checks — 5 format / capability sanity prompts
# ══════════════════════════════════════════════════════════════════════════════

SPOT_CHECKS = [
    {
        "name":            "list_flattening",
        "prompt":          "Write a Python function that flattens a nested list of arbitrary depth.",
        "expect_keywords": ["def", "flatten", "isinstance", "list", "yield"],
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
            "name":             check["name"],
            "score":            round(score, 3),
            "hits":             hits,
            "total":            total_k,
            "matched":          matched,
            "response_preview": _wrap_bare_code(response)[:400],
        })
        sym = "✓" if score >= 0.6 else "~" if score >= 0.3 else "✗"
        print(f"  {sym} {check['name']:25s}  {score:.0%}  ({hits}/{total_k})  matched={matched}")

    mean = sum(r["score"] for r in results) / len(results)
    print(f"  Mean: {mean:.1%}")
    return {"mean_score": round(mean, 3), "checks": results}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  Forge — Full Evaluation Suite  (H100 · bfloat16 · v4)")
    print("=" * 62)
    print(f"  Model      : {MERGED_MODEL_PATH}")
    print(f"  Temp={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}  rep_pen={REPEAT_PENALTY}")
    print(f"  Benchmarks : HumanEval(164) · HumanEval+ · MBPP(100) · DebugBench · Spot")
    print()

    # ── [1/6] Load model ──────────────────────────────────────────────────────
    print("[1/6] Loading model ...")
    load_model()

    # ── [2/6] Smoke test ──────────────────────────────────────────────────────
    print("[2/6] Smoke test ...")
    smoke = ask("Reply with only the word: ready", max_new_tokens=10)
    print(f"  Model says: {repr(smoke[:80])}")
    if not smoke.strip():
        print("  ✗ Empty response — check model path and load.")
        sys.exit(1)
    print("  ✓ Ready\n")

    t0 = time.time()

    # ── [3/6] HumanEval ───────────────────────────────────────────────────────
    print("[3/6] Running HumanEval (164 problems) ...")
    he = eval_humaneval(n=164)

    # ── [4/6] HumanEval+ ─────────────────────────────────────────────────────
    print("[4/6] Running HumanEval+ (extended test cases) ...")
    hep = eval_humaneval_plus(n=164)

    # ── [5/6] MBPP ───────────────────────────────────────────────────────────
    print("[5/6] Running MBPP (100 problems, function-name fix applied) ...")
    mb = eval_mbpp(n=100)

    # ── [6/6] DebugBench + Spot checks ───────────────────────────────────────
    print("[6/6] Running DebugBench + Spot checks ...")
    db = eval_debug(n=50)
    sp = eval_spot_checks()

    elapsed = time.time() - t0

    # ── Collect scores ────────────────────────────────────────────────────────
    he_score  = he.get("pass@1",       0.0)
    hep_base  = hep.get("base_pass@1", he_score)
    hep_plus  = hep.get("plus_pass@1", 0.0)
    hep_gap   = hep.get("gap",         0.0)
    mb_score  = mb.get("pass@1",       0.0)
    db_score  = db.get("accuracy",     0.0)
    sp_score  = sp.get("mean_score",   0.0)

    summary = {
        "model":                   MERGED_MODEL_PATH,
        "precision":               "bfloat16",
        "inference_mode":          "direct_hf",
        "temperature":             TEMPERATURE,
        "humaneval_pass@1":        he_score,
        "humaneval_plus_base@1":   hep_base,
        "humaneval_plus_plus@1":   hep_plus,
        "humaneval_he_vs_hep_gap": hep_gap,
        "mbpp_pass@1":             mb_score,
        "debug_accuracy":          db_score,
        "spot_mean_coverage":      sp_score,
        "eval_time_minutes":       round(elapsed / 60, 1),
        "full": {
            "humaneval":      he,
            "humaneval_plus": hep,
            "mbpp":           mb,
            "debug":          db,
            "spot":           sp,
        },
    }

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  RESULTS")
    print(f"{'=' * 62}")
    print(f"  HumanEval     pass@1      : {he_score:.2%}  ({he.get('passed',0)}/{he.get('total',164)})")
    if hep:
        print(f"  HumanEval+    base  @1    : {hep_base:.2%}")
        print(f"  HumanEval+    plus  @1    : {hep_plus:.2%}  (gap: {hep_gap:.2%})")
    else:
        print(f"  HumanEval+                : skipped  (pip install --upgrade evalplus)")
    print(f"  MBPP          pass@1      : {mb_score:.2%}  ({mb.get('passed',0)}/{mb.get('total',100)})")
    print(f"  DebugBench    accuracy    : {db_score:.2%}  ({db.get('correct',0)}/{db.get('processed',0)})")
    print(f"  Spot checks   mean        : {sp_score:.1%}")
    print(f"  Total time                : {summary['eval_time_minutes']} min")
    print(f"\n  Saved → {RESULTS_FILE}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print("  INTERPRETATION")
    print(f"{'─' * 62}")

    if hep:
        if   hep_gap <= 0.05: print("  HE vs HE+ gap ≤5%   → Model solves correctly, not by pattern-matching.")
        elif hep_gap <= 0.12: print("  HE vs HE+ gap 5-12% → Normal. Most edge cases handled.")
        else:                 print("  HE vs HE+ gap >12%  → Concern. Review failures on harder test cases.")

    if mb_score > 0:
        diff = he_score - mb_score
        if   diff <= 0.10: print("  HumanEval vs MBPP ≤10%  → Strong generalization.")
        elif diff <= 0.20: print("  HumanEval vs MBPP 10-20% → Moderate. Fine-tune data likely HumanEval-adjacent.")
        else:              print("  HumanEval vs MBPP >20%   → High gap. Add MBPP-style data to training.")
    print()


if __name__ == "__main__":
    main()
