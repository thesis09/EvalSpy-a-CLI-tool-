"""
evalspy/reporter.py
Terminal output. Uses ANSI codes directly — no rich/colorama dependency.
"""

import sys
from typing import List, Optional
from .checks.base import CheckResult, Severity, Status

# ANSI codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"
DIM    = "\033[2m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    return "".join(codes) + text + RESET


def _status_icon(status: Status, severity: Severity) -> str:
    if status == Status.PASS:
        return _c("  ✓", GREEN, BOLD)
    if status == Status.SKIP:
        return _c("  –", GRAY)
    if severity == Severity.CRITICAL:
        return _c("  ✗", RED, BOLD)
    return _c("  ~", YELLOW, BOLD)


def _severity_tag(severity: Severity) -> str:
    if severity == Severity.CRITICAL:
        return _c(" [CRITICAL]", RED, BOLD)
    if severity == Severity.WARNING:
        return _c(" [WARNING]", YELLOW)
    return _c(" [INFO]", CYAN)


def print_header(script_path: str, live: bool, server: Optional[str]) -> None:
    print()
    print(_c("  EvalSpy", BOLD, WHITE) + _c(" — LLM Evaluation Pipeline Auditor", GRAY))
    print(_c("  " + "─" * 54, GRAY))
    print(f"  {_c('Script:', BOLD)} {script_path}")
    mode = "static + live" if live else "static only"
    print(f"  {_c('Mode:', BOLD)}   {mode}")
    if live and server:
        print(f"  {_c('Server:', BOLD)}  {server}")
    print()


def print_results(results: List[CheckResult], verbose: bool = False) -> None:
    failures   = [r for r in results if r.status == Status.FAIL]
    passes     = [r for r in results if r.status == Status.PASS]
    skipped    = [r for r in results if r.status == Status.SKIP]
    critical   = [r for r in failures if r.severity == Severity.CRITICAL]
    warnings   = [r for r in failures if r.severity == Severity.WARNING]

    print(_c("  RESULTS", BOLD, WHITE))
    print(_c("  " + "─" * 54, GRAY))

    for r in results:
        icon    = _status_icon(r.status, r.severity)
        tag     = _severity_tag(r.severity) if r.status == Status.FAIL else ""
        line_ref = _c(f" (line {r.line})", GRAY) if r.line else ""
        print(f"{icon}  {_c(r.name, BOLD)}{tag}{line_ref}")
        print(f"      {_c(r.description, GRAY)}")

        if r.status == Status.FAIL and (verbose or r.severity == Severity.CRITICAL):
            if r.detail:
                for line in r.detail.strip().splitlines():
                    print(f"      {_c('│', GRAY)} {line}")
            if r.fix:
                print(f"      {_c('Fix:', BOLD + CYAN)}")
                for line in r.fix.strip().splitlines():
                    print(f"      {_c('  ' + line, CYAN)}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(_c("  " + "─" * 54, GRAY))
    print(_c("  SUMMARY", BOLD, WHITE))

    if not failures:
        print(f"  {_c('✓ All checks passed.', GREEN, BOLD)} Pipeline looks clean.")
    else:
        if critical:
            crit_names = ", ".join(r.name for r in critical)
            print(f"  {_c(f'{len(critical)} critical issue(s):', RED, BOLD)} {crit_names}")
            print(f"  {_c('  These will produce wrong benchmark scores.', RED)}")
        if warnings:
            warn_names = ", ".join(r.name for r in warnings)
            print(f"  {_c(f'{len(warnings)} warning(s):', YELLOW, BOLD)} {warn_names}")

    if skipped:
        skip_names = ", ".join(r.name for r in skipped)
        print(f"  {_c(f'{len(skipped)} skipped (benchmark not detected):', GRAY)} {skip_names}")

    score_line = (
        f"  {_c(str(len(passes)), GREEN, BOLD)} passed  "
        f"{_c(str(len(failures)), RED if failures else GRAY, BOLD)} failed  "
        f"{_c(str(len(skipped)), GRAY)} skipped"
    )
    print()
    print(score_line)
    print()

    return len(critical)
