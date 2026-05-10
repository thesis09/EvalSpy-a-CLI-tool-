"""
evalspy/checks/static.py
All AST + regex based checks. Zero ML, zero network. Runs in < 1 second.
"""

import ast
import re
from typing import List
from .base import CheckResult, Severity, Status


def _parse(source: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _line_of(source: str, match) -> int:
    return source[:match.start()].count('\n') + 1


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Duplicate function assembly (the 0% HumanEval bug)
# ─────────────────────────────────────────────────────────────────────────────

def check_assembly_logic(source: str) -> CheckResult:
    p1 = re.search(
        r'(prompt|fn_prompt|stub|function_header)\s*\+\s*["\']\\n["\']\s*\+\s*'
        r'(code|response|output|completion|extracted)',
        source, re.IGNORECASE
    )
    p2 = re.search(
        r'f["\'].*\{(prompt|fn_prompt|stub)\}.*\\n.*\{(code|response|output)\}',
        source, re.IGNORECASE
    )

    has_fn_guard = bool(re.search(
        r'(func_name|function_name|fn_name)\s+in\s+(code|response|output|completion)',
        source, re.IGNORECASE
    )) or bool(re.search(
        r'\.startswith\s*\(\s*["\']def\s', source
    ))

    if (p1 or p2) and not has_fn_guard:
        line = _line_of(source, p1 or p2)
        return CheckResult(
            name="Assembly logic",
            status=Status.FAIL,
            severity=Severity.CRITICAL,
            description="Unconditional fn_prompt prepending detected.",
            detail=(
                "When the model returns a complete function (common at temp≤0.2), "
                "prepending fn_prompt creates a duplicate 'def' block. "
                "Python silently uses the second (empty) definition. "
                "Every test fails. This produced 0% HumanEval in real evaluation runs."
            ),
            fix=(
                "Add a guard: if function_name already in model response, "
                "use response directly. Otherwise prepend stub. "
                "See: _smart_assemble() pattern."
            ),
            line=line
        )
    return CheckResult(
        name="Assembly logic",
        status=Status.PASS,
        severity=Severity.CRITICAL,
        description="No unconditional fn_prompt prepending detected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — MBPP function name injection (the 9% bug)
# ─────────────────────────────────────────────────────────────────────────────

def check_mbpp_name_injection(source: str) -> CheckResult:
    is_mbpp = bool(re.search(r'mbpp', source, re.IGNORECASE))
    if not is_mbpp:
        return CheckResult(
            name="MBPP name injection",
            status=Status.SKIP,
            severity=Severity.CRITICAL,
            description="No MBPP benchmark detected in script — check skipped."
        )

    has_name_extraction = bool(re.search(
        r'assert\s*.*re\.(search|match|findall)|'
        r'extract.*func.*name|'
        r'func_name.*assert|'
        r'test_list.*assert.*group',
        source, re.IGNORECASE
    ))
    has_name_in_prompt = bool(re.search(
        r'func_name.*prompt|named.*`.*func|must be named',
        source, re.IGNORECASE
    ))

    if not has_name_extraction or not has_name_in_prompt:
        missing = []
        if not has_name_extraction:
            missing.append("function name extraction from assert statements")
        if not has_name_in_prompt:
            missing.append("function name injection into prompt")
        return CheckResult(
            name="MBPP name injection",
            status=Status.FAIL,
            severity=Severity.CRITICAL,
            description="MBPP evaluation missing function name handling.",
            detail=(
                f"Missing: {', '.join(missing)}. "
                "MBPP test assertions hardcode the expected function name "
                "(e.g. 'assert min_cost([...]) == 4'). Without telling the model "
                "what name to use, it picks its own — causing NameError on every test. "
                "This produced 9% pass@1 in real evaluation runs (correct logic, wrong name)."
            ),
            fix=(
                "Extract name: re.search(r'assert\\s+([a-zA-Z_]\\w*)\\s*\\(', test)\n"
                "Inject name: f'Write a function named `{func_name}` that solves: {description}\\n"
                "The function MUST be named exactly `{func_name}`.'"
            )
        )
    return CheckResult(
        name="MBPP name injection",
        status=Status.PASS,
        severity=Severity.CRITICAL,
        description="Function name extraction and injection detected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — MBPP builtin name collision
# ─────────────────────────────────────────────────────────────────────────────

def check_mbpp_builtin_guard(source: str) -> CheckResult:
    is_mbpp = bool(re.search(r'mbpp', source, re.IGNORECASE))
    if not is_mbpp:
        return CheckResult(
            name="MBPP builtin guard",
            status=Status.SKIP,
            severity=Severity.WARNING,
            description="No MBPP benchmark detected — check skipped."
        )

    has_builtin_guard = bool(re.search(
        r'builtin|BUILTIN|not in.*["\']set["\']|["\']set["\']\s*not in',
        source, re.IGNORECASE
    ))

    if not has_builtin_guard:
        return CheckResult(
            name="MBPP builtin guard",
            status=Status.FAIL,
            severity=Severity.WARNING,
            description="No Python builtin exclusion in function name extraction.",
            detail=(
                "Two MBPP problems have tests like: assert set(my_function(...)) == {1,2,3}. "
                "Without a builtin guard, regex extracts 'set' as the function name. "
                "The model writes a function named 'set', shadowing Python's builtin. "
                "Both problems fail despite correct logic."
            ),
            fix=(
                "Add to extraction: \n"
                "PYTHON_BUILTINS = {'set','list','dict','tuple','int','str',\n"
                "                   'len','sum','min','max','sorted','print'}\n"
                "if match.group(1) not in PYTHON_BUILTINS: return match.group(1)"
            )
        )
    return CheckResult(
        name="MBPP builtin guard",
        status=Status.PASS,
        severity=Severity.WARNING,
        description="Builtin name exclusion detected in function name extraction."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Subprocess timeout
# ─────────────────────────────────────────────────────────────────────────────

def check_timeout(source: str) -> CheckResult:
    has_subprocess = bool(re.search(r'subprocess\.run', source))
    if not has_subprocess:
        return CheckResult(
            name="Subprocess timeout",
            status=Status.SKIP,
            severity=Severity.WARNING,
            description="No subprocess.run detected — check skipped."
        )

    no_timeout = bool(re.search(
        r'subprocess\.run\s*\([^)]*\)',
        source
    ))
    has_timeout = bool(re.search(
        r'subprocess\.run\s*\(.*timeout\s*=', source
    ))

    if no_timeout and not has_timeout:
        return CheckResult(
            name="Subprocess timeout",
            status=Status.FAIL,
            severity=Severity.WARNING,
            description="subprocess.run called without timeout parameter.",
            detail=(
                "An infinite loop in generated code will hang your evaluation forever. "
                "Without timeout, a single bad problem can stall the entire benchmark run."
            ),
            fix="Add timeout=10 (H100) or timeout=15 (local CPU/GPU) to subprocess.run()"
        )
    return CheckResult(
        name="Subprocess timeout",
        status=Status.PASS,
        severity=Severity.WARNING,
        description="Timeout parameter detected in subprocess.run."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Stop token configuration
# ─────────────────────────────────────────────────────────────────────────────

def check_stop_tokens(source: str) -> CheckResult:
    has_generation = bool(re.search(
        r'generate|create_completion|chat_completion', source, re.IGNORECASE
    ))
    if not has_generation:
        return CheckResult(
            name="Stop token config",
            status=Status.SKIP,
            severity=Severity.WARNING,
            description="No generation call detected — check skipped."
        )

    has_stop = bool(re.search(
        r'stop\s*=|eos_token|stop_sequences|end_of_turn', source, re.IGNORECASE
    ))
    has_max_tokens = bool(re.search(
        r'max_new_tokens|max_tokens', source, re.IGNORECASE
    ))

    issues = []
    if not has_stop:
        issues.append("no stop token/sequence configured")
    if not has_max_tokens:
        issues.append("no max_tokens limit set")

    if issues:
        return CheckResult(
            name="Stop token config",
            status=Status.FAIL,
            severity=Severity.WARNING,
            description=f"Generation config issues: {'; '.join(issues)}.",
            detail=(
                "Without stop tokens, the model may generate beyond the function boundary, "
                "producing code with trailing garbage that causes test failures. "
                "Without max_tokens, a runaway generation can OOM or hang."
            ),
            fix=(
                "Add: stop=['<end_of_turn>', '```', '\\n\\n\\n'] and max_new_tokens=512"
            )
        )
    return CheckResult(
        name="Stop token config",
        status=Status.PASS,
        severity=Severity.WARNING,
        description="Stop tokens and max_tokens configured."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — GGUF tokenizer corruption signatures
# ─────────────────────────────────────────────────────────────────────────────

def check_tokenizer_artifacts(source: str) -> CheckResult:
    corruption_patterns = [
        (r'for\s*,\s*in\s', "for , in  — missing loop variable"),
        (r'def\s+\w+\s*\(\s*,', "def func(,  — missing first parameter"),
        (r'UNK_BYTE', "[UNK_BYTE_*] token artifact pattern"),
        (r'\[UNK\]', "[UNK] token in output stream"),
    ]

    found = []
    for pattern, description in corruption_patterns:
        m = re.search(pattern, source)
        if m:
            found.append(f"line {_line_of(source, m)}: {description}")

    has_corruption_handler = bool(re.search(
        r'UNK_BYTE|corruption|re\.sub.*UNK', source, re.IGNORECASE
    ))

    if found and not has_corruption_handler:
        return CheckResult(
            name="Tokenizer artifact check",
            status=Status.FAIL,
            severity=Severity.CRITICAL,
            description="GGUF tokenizer corruption signatures detected in script.",
            detail=(
                f"Found patterns: {'; '.join(found)}. "
                "These are signatures of the llama-bpe hotpatch bug where variable names "
                "in Gemma's SentencePiece vocabulary decode to empty strings. "
                "Generated code will have missing identifiers silently."
            ),
            fix=(
                "1. Re-export GGUF using llama.cpp b3447+ (natively supports Gemma 3)\n"
                "2. Restore original tokenizer files before export\n"
                "3. Use chat_format=None with raw prompt strings in llama-cpp-python\n"
                "4. Add sanitizer: token = re.sub(r'\\[UNK_BYTE_[^\\]]+\\]', ' ', token)"
            )
        )

    if has_corruption_handler:
        return CheckResult(
            name="Tokenizer artifact check",
            status=Status.PASS,
            severity=Severity.CRITICAL,
            description="Corruption handler/sanitizer detected."
        )

    return CheckResult(
        name="Tokenizer artifact check",
        status=Status.PASS,
        severity=Severity.CRITICAL,
        description="No tokenizer corruption patterns detected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — DebugBench field name
# ─────────────────────────────────────────────────────────────────────────────

def check_debugbench_fields(source: str) -> CheckResult:
    is_debugbench = bool(re.search(r'debugbench|DebugBench', source))
    if not is_debugbench:
        return CheckResult(
            name="DebugBench field names",
            status=Status.SKIP,
            severity=Severity.CRITICAL,
            description="No DebugBench usage detected — check skipped."
        )

    wrong_field = bool(re.search(r'["\']fixed_code["\']', source))
    right_field = bool(re.search(r'["\']solution["\']', source))

    if wrong_field and not right_field:
        m = re.search(r'["\']fixed_code["\']', source)
        return CheckResult(
            name="DebugBench field names",
            status=Status.FAIL,
            severity=Severity.CRITICAL,
            description="Wrong field name: 'fixed_code' used instead of 'solution'.",
            detail=(
                "Rtian/DebugBench stores the corrected code under 'solution', not 'fixed_code'. "
                "Using row.get('fixed_code', '') returns '' for every row. "
                "The if not fixed_code guard then skips every sample. "
                "Result: 0 training/eval samples processed silently."
            ),
            fix="Replace 'fixed_code' with 'solution'. Or use fallback list: "
               "['solution', 'fixed_code', 'correct_code']",
            line=_line_of(source, m)
        )

    return CheckResult(
        name="DebugBench field names",
        status=Status.PASS,
        severity=Severity.CRITICAL,
        description="Correct field name 'solution' used for DebugBench."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8 — Temperature sanity
# ─────────────────────────────────────────────────────────────────────────────

def check_temperature(source: str) -> CheckResult:
    m = re.search(r'temperature\s*=\s*([0-9.]+)', source)
    if not m:
        return CheckResult(
            name="Temperature setting",
            status=Status.SKIP,
            severity=Severity.INFO,
            description="No temperature setting found."
        )

    temp = float(m.group(1))
    line = _line_of(source, m)

    if temp > 0.5:
        return CheckResult(
            name="Temperature setting",
            status=Status.FAIL,
            severity=Severity.WARNING,
            description=f"High temperature ({temp}) set for code evaluation.",
            detail=(
                f"Temperature {temp} introduces significant randomness. "
                "For code benchmarks, temperature > 0.5 causes inconsistent identifier names, "
                "malformed syntax, and non-deterministic pass@1 scores across runs. "
                "Recommended: 0.0–0.2 for deterministic code evaluation."
            ),
            fix="Set temperature=0.1 for code benchmarks (or 0.0 for fully deterministic)",
            line=line
        )

    if temp == 0.0:
        return CheckResult(
            name="Temperature setting",
            status=Status.PASS,
            severity=Severity.INFO,
            description=f"Temperature={temp} (fully deterministic — good for benchmarks).",
            line=line
        )

    return CheckResult(
        name="Temperature setting",
        status=Status.PASS,
        severity=Severity.INFO,
        description=f"Temperature={temp} (good for code evaluation).",
        line=line
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9 — Dataset assertion after loading
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset_assertions(source: str) -> CheckResult:
    has_dataset_load = bool(re.search(
        r'load_dataset|from_json|read_json|open.*jsonl', source, re.IGNORECASE
    ))
    if not has_dataset_load:
        return CheckResult(
            name="Dataset load assertions",
            status=Status.SKIP,
            severity=Severity.WARNING,
            description="No dataset loading detected — check skipped."
        )

    has_assertion = bool(re.search(
        r'assert\s+len\s*\(|assert.*>\s*0|print.*len.*samples|'
        r'logging.*len|logger.*len',
        source
    ))

    if not has_assertion:
        return CheckResult(
            name="Dataset load assertions",
            status=Status.FAIL,
            severity=Severity.WARNING,
            description="No post-load dataset size check detected.",
            detail=(
                "Silent dataset loading failures are hard to catch. "
                "A field name mismatch (e.g. 'fixed_code' vs 'solution') can cause "
                "0 samples to load with no error raised. "
                "Without an assertion, training/eval proceeds on an empty dataset."
            ),
            fix="Add after every dataset load: assert len(samples) > 0, "
               "f'No samples loaded. Available fields: {list(ds[0].keys())}'"
        )

    return CheckResult(
        name="Dataset load assertions",
        status=Status.PASS,
        severity=Severity.WARNING,
        description="Dataset size check detected after loading."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run all static checks
# ─────────────────────────────────────────────────────────────────────────────

ALL_CHECKS = [
    check_assembly_logic,
    check_mbpp_name_injection,
    check_mbpp_builtin_guard,
    check_timeout,
    check_stop_tokens,
    check_tokenizer_artifacts,
    check_debugbench_fields,
    check_temperature,
    check_dataset_assertions,
]

def run_all(source: str) -> List[CheckResult]:
    return [check(source) for check in ALL_CHECKS]
