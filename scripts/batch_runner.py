# scripts/batch_runner.py - Standalone Batch Processing Script
"""
Standalone batch processing script for running tasks without Celery
Useful for single-machine deployments or testing
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid
import traceback

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Shared Cache Bootstrap
import pathlib, torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

# App directories
for p in [
    f"{AI_CACHE_ROOT}/models/{name}"
    for name in ["lora", "blip2", "qwen", "llava", "embeddings"]
] + [f"{AI_CACHE_ROOT}/outputs/multi-modal-lab/batch"]:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

from backend.core.pipeline_loader import get_caption_pipeline, get_vqa_pipeline
from backend.utils.logging import setup_logging, get_logger
from PIL import Image

logger = get_logger(__name__)


class BatchProcessor:
    """Standalone batch processor for various tasks"""

    def __init__(self):
        self.pipelines = {}

    def get_pipeline(self, task_type: str):
        """Get or load pipeline for task type"""
        if task_type not in self.pipelines:
            if task_type == "caption":
                logger.info("Loading caption pipeline...")
                self.pipelines[task_type] = get_caption_pipeline()
            elif task_type == "vqa":
                logger.info("Loading VQA pipeline...")
                self.pipelines[task_type] = get_vqa_pipeline()
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        return self.pipelines[task_type]

    def process_caption_batch(self, items: list, batch_id: str) -> dict:
        """Process caption batch"""
        pipeline = self.get_pipeline("caption")
        results = []

        for i, item in enumerate(items):
            try:
                logger.info(f"Processing caption item {i+1}/{len(items)}")

                # Load image
                image_path = item["image_path"]
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")

                # Generate caption
                caption = pipeline(
                    image,
                    max_length=item.get("max_length", 50),
                    num_beams=item.get("num_beams", 3),
                )[0]["generated_text"]

                result = {
                    "id": item.get("id", str(uuid.uuid4())),
                    "image_path": image_path,
                    "caption": caption,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing caption item {i}: {str(e)}")
                results.append(
                    {
                        "id": item.get("id", str(uuid.uuid4())),
                        "image_path": item.get("image_path", ""),
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return {
            "batch_id": batch_id,
            "type": "caption",
            "total_items": len(items),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
            "results": results,
        }

    def process_vqa_batch(self, items: list, batch_id: str) -> dict:
        """Process VQA batch"""
        pipeline = self.get_pipeline("vqa")
        results = []

        for i, item in enumerate(items):
            try:
                logger.info(f"Processing VQA item {i+1}/{len(items)}")

                # Load image and question
                image_path = item["image_path"]
                question = item["question"]

                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")

                # Generate answer
                answer = pipeline(
                    image, question, max_length=item.get("max_length", 100)
                )

                result = {
                    "id": item.get("id", str(uuid.uuid4())),
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing VQA item {i}: {str(e)}")
                results.append(
                    {
                        "id": item.get("id", str(uuid.uuid4())),
                        "image_path": item.get("image_path", ""),
                        "question": item.get("question", ""),
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return {
            "batch_id": batch_id,
            "type": "vqa",
            "total_items": len(items),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "failed"]),
            "results": results,
        }

    def run_batch(
        self, task_type: str, input_file: str, output_file: str = None
    ) -> str:
        """Run batch processing"""
        try:
            # Generate batch ID
            batch_id = str(uuid.uuid4())

            # Load input data
            if input_file.endswith(".csv"):
                df = pd.read_csv(input_file)
                items = df.to_dict("records")
            elif input_file.endswith(".json"):
                with open(input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else data.get("items", [])
            else:
                raise ValueError("Input file must be CSV or JSON format")

            logger.info(f"Starting batch {batch_id} with {len(items)} items")

            # Process batch
            if task_type == "caption":
                result_data = self.process_caption_batch(items, batch_id)
            elif task_type == "vqa":
                result_data = self.process_vqa_batch(items, batch_id)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Save results
            if output_file is None:
                output_dir = f"{AI_CACHE_ROOT}/outputs/multi-modal-lab/batch"
                output_file = f"{output_dir}/{task_type}_batch_{batch_id}.json"

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            result_data["completed_at"] = datetime.now().isoformat()

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Batch completed. Results saved to: {output_file}")
            logger.info(
                f"Summary: {result_data['successful']}/{result_data['total_items']} successful"
            )

            return output_file

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}\n{traceback.format_exc()}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Standalone batch processor")
    parser.add_argument(
        "task_type", choices=["caption", "vqa"], help="Task type to process"
    )
    parser.add_argument("input_file", help="Input CSV or JSON file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    if args.verbose:
        logger.setLevel("DEBUG")

    # Set device
    if args.device != "auto":
        os.environ["DEVICE"] = args.device

    # Run batch processing
    processor = BatchProcessor()
    try:
        output_file = processor.run_batch(args.task_type, args.input_file, args.output)
        print(f"✅ Batch processing completed successfully!")
        print(f"📄 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
