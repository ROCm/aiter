#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse workflow files and output job matrix for GitHub Actions."
    )
    parser.add_argument(
        "--workflow-dir",
        default=".github/workflows",
        help="Directory containing workflow yaml files.",
    )
    parser.add_argument(
        "--workflows",
        default="",
        help="Comma-separated workflow file names. Empty means auto-discover.",
    )
    parser.add_argument(
        "--exclude-workflows",
        default="",
        help="Comma-separated workflow file names to skip.",
    )
    parser.add_argument(
        "--exclude-jobs",
        default="",
        help="Comma-separated job names to skip.",
    )
    parser.add_argument(
        "--out-matrix", required=True, help="Output path for matrix JSON."
    )
    parser.add_argument(
        "--out-workflow-map",
        required=True,
        help="Output path for workflow -> jobs mapping JSON.",
    )
    return parser.parse_args()


def parse_csv(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def discover_workflows(workflow_dir: Path):
    return sorted(
        [path.name for path in workflow_dir.glob("*.yml") if path.is_file()]
        + [path.name for path in workflow_dir.glob("*.yaml") if path.is_file()]
    )


def main():
    args = parse_args()
    workflow_dir = Path(args.workflow_dir)
    excluded_workflows = set(parse_csv(args.exclude_workflows))
    excluded_jobs = set(parse_csv(args.exclude_jobs))

    if args.workflows.strip():
        workflow_files = parse_csv(args.workflows)
    else:
        workflow_files = discover_workflows(workflow_dir)

    workflow_files = [wf for wf in workflow_files if wf not in excluded_workflows]

    matrix = []
    workflow_map = {}

    for workflow_file in workflow_files:
        workflow_path = workflow_dir / workflow_file
        if not workflow_path.exists():
            continue

        with workflow_path.open("r", encoding="utf-8") as file_obj:
            content = yaml.safe_load(file_obj) or {}

        jobs_dict = content.get("jobs") or {}
        job_ids = [job_id for job_id in jobs_dict.keys() if job_id not in excluded_jobs]

        display_names = []
        for job_id in job_ids:
            job_def = jobs_dict.get(job_id) or {}
            raw_name = job_def.get("name") if isinstance(job_def, dict) else None
            if isinstance(raw_name, str) and "${{" not in raw_name:
                display_name = raw_name
            else:
                display_name = job_id
            display_names.append(display_name)
            matrix.append(
                {
                    "workflow": workflow_file,
                    "job_id": job_id,
                    "job_name": display_name,
                }
            )

        workflow_map[workflow_file] = display_names

    Path(args.out_matrix).write_text(
        json.dumps(matrix, ensure_ascii=False), encoding="utf-8"
    )
    Path(args.out_workflow_map).write_text(
        json.dumps(workflow_map, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Discovered workflows: {len(workflow_map)}")
    print(f"Total jobs in matrix: {len(matrix)}")


if __name__ == "__main__":
    main()
