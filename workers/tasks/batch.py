# workers/tasks/batch.py - Batch processing tasks
from workers.celery_app import celery_app
from workers.tasks.generation import generate_single_image_task
from typing import Dict, List, Any
import json
from datetime import datetime


@celery_app.task(bind=True, name="batch_generate")
def batch_generate_task(
    self,
    batch_id: str,
    prompts: List[str],
    common_params: Dict[str, Any],
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Process batch generation job"""

    try:
        total_prompts = len(prompts)
        completed = 0
        failed = 0
        results = []

        # Process prompts in batches
        for i in range(0, total_prompts, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_results = []

            # Process each prompt in current batch
            for j, prompt in enumerate(batch_prompts):
                try:
                    # Update progress
                    current_task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": completed + j + 1,
                            "total": total_prompts,
                            "batch_id": batch_id,
                        },
                    )

                    # Generate image
                    result = generate_single_image_task.delay(
                        prompt=prompt, **common_params
                    ).get()  # Synchronous execution within batch

                    batch_results.append(
                        {"prompt_index": i + j, "prompt": prompt, "result": result}
                    )

                    if result["status"] == "success":
                        completed += 1
                    else:
                        failed += 1

                except Exception as e:
                    failed += 1
                    batch_results.append(
                        {
                            "prompt_index": i + j,
                            "prompt": prompt,
                            "result": {"status": "failed", "error": str(e)},
                        }
                    )

            results.extend(batch_results)

            # Small delay between batches to prevent overload
            time.sleep(1)

        # Save batch results
        cache_paths = get_cache_paths()
        batch_dir = cache_paths.outputs / "batch" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        results_file = batch_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "batch_id": batch_id,
                    "total_prompts": total_prompts,
                    "completed": completed,
                    "failed": failed,
                    "results": results,
                    "completed_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        return {
            "status": "completed",
            "batch_id": batch_id,
            "total_prompts": total_prompts,
            "completed": completed,
            "failed": failed,
            "results_file": str(results_file),
        }

    except Exception as e:
        return {
            "status": "failed",
            "batch_id": batch_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
