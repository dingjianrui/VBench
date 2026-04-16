"""
Sample videos for VBench-2.0 evaluation using Volcengine Ark API.

Usage:
    python sample_videos.py \
        --save_path ./sampled_videos \
        --api_key YOUR_ARK_API_KEY \
        [--model doubao-seedance-1-0-pro-250528] \
        [--prompt_file ./prompts/VBench2_full_text.txt] \
        [--max_parallel 10] \
        [--poll_interval 10] \
        [--resume]
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"

# First 10 prompts in VBench2_full_text.txt are Diversity (need 20 videos each),
# the rest need 3 videos each.
DIVERSITY_PROMPT_COUNT = 10
DIVERSITY_VIDEO_COUNT = 20
DEFAULT_VIDEO_COUNT = 3


def load_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def build_video_tasks(prompts: list[str]) -> list[dict]:
    """Build list of (prompt, index, save_name) tasks."""
    tasks = []
    for idx, prompt in enumerate(prompts):
        n_videos = DIVERSITY_VIDEO_COUNT if idx < DIVERSITY_PROMPT_COUNT else DEFAULT_VIDEO_COUNT
        for video_idx in range(n_videos):
            save_name = f"{prompt[:180]}-{video_idx}.mp4"
            tasks.append({
                "prompt": prompt,
                "index": video_idx,
                "save_name": save_name,
            })
    return tasks


def load_progress(progress_file: str) -> set[str]:
    """Load set of already-completed save_names."""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(line.strip() for line in f if line.strip())


def record_progress(progress_file: str, save_name: str):
    with open(progress_file, "a") as f:
        f.write(save_name + "\n")


async def create_task(
    session: aiohttp.ClientSession,
    headers: dict,
    prompt: str,
    model: str,
) -> str | None:
    """Submit a video generation task. Returns task_id or None on failure."""
    payload = {
        "model": model,
        "content": [{"type": "text", "text": prompt}],
        "seed": -1,
        "generate_audio": False,
        "resolution": "720p",
        "ratio": "16:9",
        "duration": 5,
        "watermark": False,
    }
    try:
        async with session.post(BASE_URL, headers=headers, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error("Create task failed (HTTP %d): %s", resp.status, data)
                return None
            task_id = data.get("id")
            if not task_id:
                logger.error("Create task response missing 'id': %s", data)
                return None
            return task_id
    except Exception as e:
        logger.error("Create task request error: %s", e)
        return None


async def poll_task(
    session: aiohttp.ClientSession,
    headers: dict,
    task_id: str,
    poll_interval: int,
) -> dict | None:
    """Poll task until terminal status. Returns response dict or None."""
    terminal_statuses = {"succeeded", "failed", "cancelled", "expired"}
    while True:
        try:
            async with session.get(f"{BASE_URL}/{task_id}", headers=headers) as resp:
                data = await resp.json()
                status = data.get("status")
                if status in terminal_statuses:
                    return data
                logger.debug("Task %s status: %s", task_id, status)
        except Exception as e:
            logger.error("Poll task %s error: %s", task_id, e)
        await asyncio.sleep(poll_interval)


async def download_video(
    session: aiohttp.ClientSession,
    video_url: str,
    save_path: str,
):
    """Download video from URL to local file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        async with session.get(video_url) as resp:
            if resp.status != 200:
                logger.error("Download failed (HTTP %d) for %s", resp.status, save_path)
                return False
            with open(save_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        logger.error("Download error for %s: %s", save_path, e)
        return False


async def process_one_task(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    headers: dict,
    task_info: dict,
    model: str,
    save_dir: str,
    poll_interval: int,
    progress_file: str,
    max_retries: int = 3,
) -> bool:
    """Create, poll, and download one video. Respects concurrency semaphore."""
    async with sem:
        save_path = os.path.join(save_dir, task_info["save_name"])
        prompt = task_info["prompt"]
        label = f"[{task_info['save_name']}]"

        for attempt in range(1, max_retries + 1):
            logger.info("%s Creating task (attempt %d/%d)...", label, attempt, max_retries)
            task_id = await create_task(session, headers, prompt, model)
            if not task_id:
                logger.warning("%s Failed to create task, retrying...", label)
                await asyncio.sleep(5)
                continue

            logger.info("%s Task created: %s, polling...", label, task_id)
            result = await poll_task(session, headers, task_id, poll_interval)
            if result is None:
                logger.warning("%s Poll returned None, retrying...", label)
                continue

            status = result.get("status")
            if status == "succeeded":
                video_url = result.get("content", {}).get("video_url")
                if not video_url:
                    logger.error("%s Succeeded but no video_url in response", label)
                    continue
                logger.info("%s Downloading video...", label)
                ok = await download_video(session, video_url, save_path)
                if ok:
                    record_progress(progress_file, task_info["save_name"])
                    logger.info("%s Done.", label)
                    return True
                else:
                    logger.warning("%s Download failed, retrying...", label)
                    continue
            else:
                logger.warning("%s Task ended with status: %s, retrying...", label, status)
                continue

        logger.error("%s All %d attempts failed.", label, max_retries)
        return False


async def main(args):
    prompts = load_prompts(args.prompt_file)
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompt_file)

    all_tasks = build_video_tasks(prompts)
    total = len(all_tasks)
    logger.info("Total video tasks: %d", total)

    os.makedirs(args.save_path, exist_ok=True)
    progress_file = os.path.join(args.save_path, ".progress.txt")

    if args.resume:
        done = load_progress(progress_file)
        before = len(all_tasks)
        all_tasks = [t for t in all_tasks if t["save_name"] not in done]
        logger.info("Resume mode: skipping %d already completed, %d remaining", before - len(all_tasks), len(all_tasks))
    else:
        # Clear progress file on fresh start
        if os.path.exists(progress_file):
            os.remove(progress_file)

    if not all_tasks:
        logger.info("All tasks already completed!")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    sem = asyncio.Semaphore(args.max_parallel)

    async with aiohttp.ClientSession() as session:
        coros = [
            process_one_task(
                sem, session, headers, task_info,
                args.model, args.save_path, args.poll_interval, progress_file,
            )
            for task_info in all_tasks
        ]
        results = await asyncio.gather(*coros)

    succeeded = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    logger.info("Completed: %d succeeded, %d failed out of %d total", succeeded, failed, len(results))


def parse_args():
    parser = argparse.ArgumentParser(description="Sample videos for VBench-2.0 via Volcengine Ark API")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save videos")
    parser.add_argument("--api_key", type=str, default=os.environ.get("ARK_API_KEY", ""),
                        help="Ark API key (or set ARK_API_KEY env var)")
    parser.add_argument("--model", type=str, default="doubao-seedance-2-0-260128",
                        help="Model name for video generation")
    parser.add_argument("--prompt_file", type=str, default="./prompts/VBench2_full_text.txt",
                        help="Path to prompt list file")
    parser.add_argument("--max_parallel", type=int, default=10,
                        help="Max parallel video generation tasks (default: 10)")
    parser.add_argument("--poll_interval", type=int, default=10,
                        help="Seconds between polling task status (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run, skipping completed videos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise ValueError("API key required. Pass --api_key or set ARK_API_KEY env var.")
    asyncio.run(main(args))
