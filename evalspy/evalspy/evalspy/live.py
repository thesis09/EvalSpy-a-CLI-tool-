"""
evalspy/live.py
Optional live mode: runs 5 sample problems against your actual model
and catches runtime failures that static analysis cannot see.
"""

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

from .checks.base import CheckResult, Severity, Status


LIVE_HUMANEVAL_SAMPLES = [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
            '    """ Check if in given list of numbers, are any two numbers closer to each other\n'
            '    than given threshold.\n'
            '    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n'
            '    False\n'
            '    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n'
            '    True\n'
            '    """\n'
        ),
        "test": (
            "def check(has_close_elements):\n"
            "    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
            "    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
            "check(has_close_elements)\n"
        )
    },
    {
        "task_id": "HumanEval/1",
        "prompt": (
            'def separate_paren_groups(paren_string: str) -> List[str]:\n'
            '    """ Input to this function is a string containing multiple groups of nested parentheses.\n'
            '    Your goal is to separate those group into separate strings and return the list of those.\n'
            '    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n'
            '    Ignore any spaces in the input string.\n'
            '    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n'
            '    [\'()\', \'(())\', \'(()())\']\n'
            '    """\n'
        ),
        "test": (
            "def check(separate_paren_groups):\n"
            "    assert separate_paren_groups('(()()) ((())) () ((())(()))') == ['(()())', '((()))', '()', '((())(()))']\n"
            "    assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n"
            "check(separate_paren_groups)\n"
        )
    },
    {
        "task_id": "HumanEval/2",
        "prompt": (
            'def truncate_number(number: float) -> float:\n'
            '    """ Given a positive floating point number, it can be decomposed into\n'
            '    and integer part (largest integer smaller than given number) and decimals\n'
            '    (leftover part always smaller than 1).\n'
            '    Return the decimal part of the number.\n'
            '    >>> truncate_number(3.5)\n'
            '    0.5\n'
            '    """\n'
        ),
        "test": (
            "def check(truncate_number):\n"
            "    assert abs(truncate_number(3.5) - 0.5) < 1e-6\n"
            "    assert abs(truncate_number(1.33) - 0.33) < 1e-6\n"
            "    assert abs(truncate_number(123.456) - 0.456) < 1e-6\n"
            "check(truncate_number)\n"
        )
    },
]


def _ask_model(server_url: str, fn_prompt: str, timeout: int = 60) -> str:
    import urllib.request
    payload = json.dumps({
        "messages": [{"role": "user", "content": (
            "Complete this Python function. "
            "Return ONLY the completed function inside a ```python``` block.\n\n"
            f"```python\n{fn_prompt}\n```"
        )}],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def _extract_code(response: str, fn_prompt: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    code = m.group(1).strip() if m else response.strip()

    fn_name_match = re.search(r"def\s+(\w+)\s*\(", fn_prompt)
    if fn_name_match and f"def {fn_name_match.group(1)}" in code:
        return code

    if not code.lstrip().startswith("def "):
        indented = "\n".join(
            ("    " + line) if line.strip() else ""
            for line in code.splitlines()
        )
        return fn_prompt.rstrip() + "\n" + indented

    return fn_prompt.rstrip() + "\n    pass"


def _run_code(code: str, test: str) -> Tuple[bool, str]:
    full = code + "\n\n" + test
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        res = subprocess.run(
            [sys.executable, fname],
            capture_output=True, timeout=10
        )
        if res.returncode == 0:
            return True, ""
        return False, res.stderr.decode()[:300]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT — infinite loop in generated code"
    except Exception as e:
        return False, str(e)
    finally:
        Path(fname).unlink(missing_ok=True)


def run_live_check(server_url: str) -> CheckResult:
    passed = 0
    details = []

    for sample in LIVE_HUMANEVAL_SAMPLES:
        task_id   = sample["task_id"]
        fn_prompt = sample["prompt"]
        test_code = sample["test"]

        response  = _ask_model(server_url, fn_prompt)
        if response.startswith("ERROR:"):
            return CheckResult(
                name="Live model check",
                status=Status.FAIL,
                severity=Severity.CRITICAL,
                description=f"Could not connect to model server at {server_url}.",
                detail=response,
                fix=f"Start your server first: python main.py --model your.gguf\nExpected at: {server_url}"
            )

        code = _extract_code(response, fn_prompt)
        ok, err = _run_code(code, test_code)

        if ok:
            passed += 1
            details.append(f"  ✓ {task_id}")
        else:
            first_line = err.splitlines()[0] if err else "unknown error"
            details.append(f"  ✗ {task_id} — {first_line}")

    score = passed / len(LIVE_HUMANEVAL_SAMPLES)
    detail_str = "\n".join(details)

    if score == 1.0:
        return CheckResult(
            name="Live model check",
            status=Status.PASS,
            severity=Severity.INFO,
            description=f"All {len(LIVE_HUMANEVAL_SAMPLES)} live sample problems passed.",
            detail=detail_str
        )
    elif score >= 0.6:
        return CheckResult(
            name="Live model check",
            status=Status.FAIL,
            severity=Severity.WARNING,
            description=f"Live check: {passed}/{len(LIVE_HUMANEVAL_SAMPLES)} passed.",
            detail=detail_str,
            fix="Check your assembly logic and stop token configuration."
        )
    else:
        return CheckResult(
            name="Live model check",
            status=Status.FAIL,
            severity=Severity.CRITICAL,
            description=f"Live check: {passed}/{len(LIVE_HUMANEVAL_SAMPLES)} passed — likely pipeline bug.",
            detail=detail_str,
            fix="Run with --verbose to see generated code. Check assembly logic first."
        )
