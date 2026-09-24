"""Run the Part 1 project checks and print a PASS/FAIL report.

Usage:
    python check.py           # all checks: subsystem sign tests + Simulations 1-4
    python check.py --fast    # subsystem sign tests only (seconds)

The same checks run via ``pytest`` and in ``notebooks/part_1_demo.ipynb``
(where every simulation is also plotted); see ``simulation/checks.py`` for
what each check verifies.  The exit code is non-zero unless everything
passes.  The CI workflow (``.github/workflows/check.yml``) runs ``pytest``,
which reports unimplemented subsystems as skipped instead of failed, so a
fresh template starts green on GitHub.
"""
import argparse
import sys

from simulation.checks import ALL_CHECKS, FAST_CHECKS, plot_check_results, print_report, run_all, run_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="run only the fast subsystem checks")
    parser.add_argument("--check", "--sim", "-c", "-s", nargs="+",
                        choices=sorted(ALL_CHECKS),
                        help=("run only the named checks, e.g. current, wind, "
                              "sim1_current, sim3_setpoint_change"))
    parser.add_argument("--show-plots", action="store_true",
                        help="display the saved simulation plots after the run")
    args = parser.parse_args()

    if args.check:
        results = [run_check(key) for key in args.check]
    elif args.fast:
        results = [run_check(key) for key in FAST_CHECKS]
    else:
        results = run_all(fast_only=False)

    report_ok = print_report(results)
    if not args.fast or args.check:
        plot_check_results(results, show=args.show_plots)
    return 0 if report_ok else 1


if __name__ == "__main__":
    sys.exit(main())
