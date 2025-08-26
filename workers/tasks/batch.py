# workers/tasks/batch.py - Batch processing tasks
import csv
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import traceback
import time
import logging

from celery import current_task
from workers.celery_app import celery_app
from workers.tasks.generation import generate_single_image_task
from core.config import get_cache_paths

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="batch_generate")
def batch_generate_task(
    self,
    batch_id: str,
    prompts: List[str],
    common_params: Dict[str, Any],
    batch_size: int = 4,
    save_progress: bool = True,
) -> Dict[str, Any]:
    """Process batch generation job with progress tracking"""

    try:
        logger.info(
            f"Starting batch generation: {batch_id} with {len(prompts)} prompts"
        )

        total_prompts = len(prompts)
        completed = 0
        failed = 0
        results = []

        # Create batch output directory
        cache_paths = get_cache_paths()
        batch_dir = cache_paths.outputs / "batch" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save batch info
        batch_info = {
            "batch_id": batch_id,
            "total_prompts": total_prompts,
            "common_params": common_params,
            "started_at": datetime.now().isoformat(),
        }

        with open(batch_dir / "batch_info.json", "w") as f:
            json.dump(batch_info, f, indent=2)

        # Process prompts in batches
        for i in range(0, total_prompts, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_results = []

            # Process each prompt in current batch
            for j, prompt in enumerate(batch_prompts):
                prompt_index = i + j

                try:
                    # Update progress
                    current_task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": completed + 1,
                            "total": total_prompts,
                            "batch_id": batch_id,
                            "current_prompt": (
                                prompt[:50] + "..." if len(prompt) > 50 else prompt
                            ),
                        },
                    )

                    # Add prompt index to seed for variation
                    generation_params = common_params.copy()
                    if "seed" in generation_params:
                        generation_params["seed"] = (
                            generation_params["seed"] + prompt_index
                        )

                    # Generate image
                    result = generate_single_image_task(
                        prompt=prompt, **generation_params
                    )

                    batch_results.append(
                        {
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "result": result,
                            "processed_at": datetime.now().isoformat(),
                        }
                    )

                    if result["status"] == "success":
                        completed += 1
                        logger.info(
                            f"Generated image {completed}/{total_prompts}: {prompt[:30]}..."
                        )
                    else:
                        failed += 1
                        logger.warning(
                            f"Failed generation {prompt_index}: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    failed += 1
                    logger.error(f"Exception in prompt {prompt_index}: {e}")
                    batch_results.append(
                        {
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "result": {"status": "failed", "error": str(e)},
                            "processed_at": datetime.now().isoformat(),
                        }
                    )

            results.extend(batch_results)

            # Save intermediate progress
            if save_progress:
                progress_file = batch_dir / f"progress_{i // batch_size + 1}.json"
                with open(progress_file, "w") as f:
                    json.dump(batch_results, f, indent=2)

            # Small delay between batches to prevent overload
            time.sleep(2)

        # Create summary
        summary = {
            "batch_id": batch_id,
            "total_prompts": total_prompts,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_prompts if total_prompts > 0 else 0,
            "completed_at": datetime.now().isoformat(),
        }

        # Save final results
        final_results = {**summary, "results": results}

        results_file = batch_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        # Create CSV summary for easy viewing
        csv_file = batch_dir / "summary.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "Prompt", "Status", "Image Path", "Error"])

            for item in results:
                result = item["result"]
                writer.writerow(
                    [
                        item["prompt_index"],
                        item["prompt"][:100],
                        result["status"],
                        result.get("image_path", ""),
                        result.get("error", ""),
                    ]
                )

        logger.info(
            f"Batch generation completed: {completed}/{total_prompts} successful"
        )

        return {
            **summary,
            "results_file": str(results_file),
            "csv_file": str(csv_file),
            "output_directory": str(batch_dir),
        }

    except Exception as e:
        error_msg = f"Batch generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return {
            "status": "failed",
            "batch_id": batch_id,
            "error": error_msg,
            "traceback": traceback.format_exc(),
        }


@celery_app.task(name="batch_from_csv")
def batch_from_csv_task(
    batch_id: str, csv_file_path: str, common_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Process batch from CSV file"""

    try:
        # Read CSV file
        prompts = []
        csv_path = Path(csv_file_path)

        if not csv_path.exists():
            return {"status": "failed", "error": f"CSV file not found: {csv_file_path}"}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Assume first column is prompt, or look for 'prompt' column
                if "prompt" in row:
                    prompt = row["prompt"]
                else:
                    # Use first column
                    prompt = list(row.values())[0] if row else ""

                if prompt.strip():
                    prompts.append(prompt.strip())

        if not prompts:
            return {"status": "failed", "error": "No valid prompts found in CSV file"}

        logger.info(f"Loaded {len(prompts)} prompts from CSV: {csv_file_path}")

        # Start batch generation
        return batch_generate_task(batch_id, prompts, common_params)  # type: ignore

    except Exception as e:
        return {"status": "failed", "error": f"CSV batch processing failed: {str(e)}"}
