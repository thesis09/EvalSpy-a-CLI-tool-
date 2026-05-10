"""
evalspy/cli.py
Entry point. Pure stdlib — no click, no rich.
"""

import argparse
import sys
from pathlib import Path

from .checks.static import run_all
from .reporter import print_header, print_results


def main():
    parser = argparse.ArgumentParser(
        prog="evalspy",
        description="Audit your LLM benchmark eval pipeline for known failure modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evalspy check my_eval.py
  evalspy check my_eval.py --verbose
  evalspy check my_eval.py --live --server http://localhost:8080
  evalspy check my_eval.py --benchmark humaneval mbpp debugbench
        """
    )

    sub = parser.add_subparsers(dest="command")

    # ── check sub-command ────────────────────────────────────────────────────
    check_parser = sub.add_parser("check", help="Audit an evaluation script")
    check_parser.add_argument("script", help="Path to your eval .py file")
    check_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show full details for all failures, including warnings"
    )
    check_parser.add_argument(
        "--live", "-l", action="store_true",
        help="Run 3 live sample problems against your model server"
    )
    check_parser.add_argument(
        "--server", "-s", default="http://127.0.0.1:8080",
        help="Model server URL for --live mode (default: http://127.0.0.1:8080)"
    )
    check_parser.add_argument(
        "--benchmark", "-b", nargs="+",
        choices=["humaneval", "mbpp", "debugbench", "all"],
        default=["all"],
        help="Which benchmark checks to focus on"
    )
    check_parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (for CI integration)"
    )

    # ── list sub-command ─────────────────────────────────────────────────────
    list_parser = sub.add_parser("list", help="List all available checks")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        # Run on empty string to get check names
        sample_results = run_all("")
        print("\n  Available checks (9 total):\n")
        descriptions = {
            "Assembly logic":        "Duplicate def blocks → 0% HumanEval",
            "MBPP name injection":   "Missing function name → NameError → 9% MBPP",
            "MBPP builtin guard":    "set() collision masking real function name",
            "Subprocess timeout":    "Infinite loops hanging benchmark run",
            "Stop token config":     "Model generating past function boundary",
            "Tokenizer artifact check": "GGUF variable name corruption patterns",
            "DebugBench field names":"fixed_code vs solution field mismatch → 0 samples",
            "Temperature setting":   "High temp causing non-deterministic scores",
            "Dataset load assertions":"Silent 0-sample dataset loads",
        }
        for r in sample_results:
            desc = descriptions.get(r.name, r.description)
            sev  = f"[{r.severity.value}]"
            print(f"  • {r.name:<30} {sev:<12} {desc}")
        print()
        sys.exit(0)

    if args.command == "check":
        path = Path(args.script)
        if not path.exists():
            print(f"\n  ✗ File not found: {path}\n")
            sys.exit(1)

        source = path.read_text(encoding="utf-8")

        print_header(
            script_path=str(path),
            live=args.live,
            server=args.server if args.live else None
        )

        # Static checks
        results = run_all(source)

        # Live check
        if args.live:
            from .live import run_live_check
            print("  Running live check against model server...")
            live_result = run_live_check(args.server)
            results.append(live_result)

        # JSON output for CI
        if args.json:
            import json
            output = [
                {
                    "name":     r.name,
                    "status":   r.status.value,
                    "severity": r.severity.value,
                    "description": r.description,
                    "detail":   r.detail,
                    "fix":      r.fix,
                    "line":     r.line,
                }
                for r in results
            ]
            print(json.dumps(output, indent=2))
            critical = sum(1 for r in results
                           if r.status.value == "FAIL" and r.severity.value == "CRITICAL")
            sys.exit(1 if critical else 0)

        critical_count = print_results(results, verbose=args.verbose)
        sys.exit(1 if critical_count else 0)


if __name__ == "__main__":
    main()
