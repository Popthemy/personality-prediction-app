"""Django wrapper for the Django-free research-contract readiness gate."""

from django.core.management.base import BaseCommand, CommandError

from backend.ml_pipeline.experiments.research_contract import (
    format_report,
    validate_research_contract,
)


class Command(BaseCommand):
    help = (
        "Validate the PANDORA research contract (reproducibility, data quality, "
        "prediction quality, interpretation, experiment design, leakage). "
        "WARNING does not stop execution; FAIL names the exact missing condition."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "path",
            nargs="?",
            default=None,
            help="Run output directory, run_summary.json, or multi_seed_summary.json",
        )
        parser.add_argument(
            "--json",
            dest="json_out",
            default=None,
            help="Write the machine-readable report to this path",
        )
        parser.add_argument(
            "--skip-source",
            action="store_true",
            help="Skip static runner-source leakage checks",
        )

    def handle(self, *args, **options):
        report = validate_research_contract(
            options.get("path"),
            include_source_checks=not options.get("skip_source"),
        )
        self.stdout.write(format_report(report))
        json_out = options.get("json_out")
        if json_out:
            import json
            from pathlib import Path

            path = Path(json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if report.get("status") == "FAIL":
            raise CommandError(
                "Research contract FAIL. See missing/invalid conditions above."
            )
        if report.get("status") == "WARNING":
            self.stdout.write(self.style.WARNING("Research contract WARNING (execution not stopped)."))
        else:
            self.stdout.write(self.style.SUCCESS("Research contract PASS."))
