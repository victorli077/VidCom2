import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger as eval_logger

DEFAULT_PRE_PROMPT = ""
DEFAULT_POST_PROMPT = "Respond with only the letter (A, B, C, or D) of the best answer."

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")
CACHE_ROOT = os.path.expanduser(os.getenv("DAILYOMNI_CACHE", os.path.join(HF_HOME, "Daily-Omni")))


def dailyomni_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter (A/B/C/D) from the Answer field."""
    answer = doc.get("Answer", "")
    if isinstance(answer, str):
        return answer.strip().upper()
    return str(answer).upper()


def dailyomni_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build the textual prompt for Daily-Omni audio-visual video understanding.

    Dataset columns:
      - Question: str
      - Choice: list of 4 strings, each like "A. Option text"
      - Answer: 'A'/'B'/'C'/'D'
      - video_id: str (YouTube video ID)
      - Type: str (task type: AV Event Alignment, Event Sequence, Inference, etc.)
      - content_parent_category: str
      - content_fine_category: str
      - video_category: str
      - video_duration: str (e.g. "30s")
      - Explanation: str or null
    """
    pre_prompt = DEFAULT_PRE_PROMPT
    post_prompt = DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    raw_choices = doc.get("Choice", [])
    if isinstance(raw_choices, np.ndarray):
        raw_choices = raw_choices.tolist()

    prompt_parts = [
        pre_prompt,
        "Select the best answer to the following multiple-choice question based on the video. "
        "You should carefully watch the video and listen to the audio to determine the correct answer. "
        "Respond with only the letter (A, B, C, or D) of the best option.\n",
        doc.get("Question", ""),
        "\n",
    ]

    for choice in raw_choices:
        prompt_parts.append(str(choice))
        prompt_parts.append("\n")

    prompt_parts.append(post_prompt)
    return "".join(prompt_parts).strip()


def dailyomni_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """
    Return the video clip path resolved from the hub cache.

    The video_id field is a YouTube video ID (e.g. "Ec_lQgZ9wlg").
    The tar archive extracts to this structure:
        {CACHE_ROOT}/Videos/{video_id}/{video_id}_video.mp4
    e.g. /.../Daily-Omni/Videos/Ec_lQgZ9wlg/Ec_lQgZ9wlg_video.mp4
    """
    video_id = doc.get("video_id", "")
    if not video_id:
        return []

    full_path = os.path.join(CACHE_ROOT, "Videos", video_id, f"{video_id}_video.mp4")
    full_path = os.path.expanduser(full_path)
    return [full_path]


def parse_multi_choice_response(response: Optional[str], all_choices: List[str]) -> str:
    """
    Parse the model text output into a choice letter (A/B/C/D).

    Args:
        response: Raw model string output.
        all_choices: Valid labels, e.g. ["A", "B", "C", "D"].

    Returns:
        The parsed letter (uppercased). Falls back to the first choice if nothing matches.
    """
    if response is None:
        response = ""

    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "Based",
        "Correct answer",
        "\u261e",
        "<|im_end|>",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    response = response.strip()
    response = re.sub(r"[.,:!\"'`;\\/?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]", " ", response)
    tokens = response.split()

    for token in tokens:
        if token in all_choices or token.upper() in all_choices:
            return token.upper()

    return all_choices[0]


def dailyomni_process_results(
    doc: Dict[str, Any], results: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare model prediction with ground truth for Daily-Omni benchmark.

    Args:
        doc: Single dataset example.
        results: List of model output strings for this example.

    Returns:
        Dict keyed by metric name with per-sample fields used in aggregation.
        Aggregates by Type (task type), content_parent_category, and video_duration (30s/60s) for detailed breakdown.
    """
    pred = results[0] if results else ""
    all_choices = ["A", "B", "C", "D"]
    answer = parse_multi_choice_response(pred, all_choices)
    gt_answer = dailyomni_doc_to_target(doc)
    score = 1.0 if answer == gt_answer else 0.0

    return {
        "dailyomni_score": {
            "question_id": doc.get("video_id", ""),
            "task_type": doc.get("Type", "unknown"),
            "content_category": doc.get("content_parent_category") or "unknown",
            "video_duration": doc.get("video_duration", "unknown"),
            "score": score,
        }
    }


def dailyomni_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate per-sample scores into per-task-type and per-content-category accuracy,
    plus overall accuracy.

    This is executed in the main process (rank 0) after all samples are processed.

    Args:
        results: List of dicts produced by dailyomni_process_results.

    Returns:
        Overall accuracy (percentage 0-100).
    """
    task_type_scores = defaultdict(list)
    content_cat_scores = defaultdict(list)
    duration_scores = defaultdict(list)
    total_score = 0.0

    for result in results:
        task_type = result.get("task_type", "unknown")
        content_cat = result.get("content_category") or "unknown"
        duration = result.get("video_duration") or "unknown"
        score = float(result.get("score", 0.0))

        task_type_scores[task_type].append(score)
        content_cat_scores[content_cat].append(score)
        duration_scores[duration].append(score)
        total_score += score

    n = len(results)
    overall = (total_score / n * 100.0) if n > 0 else 0.0

    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall Daily-Omni Accuracy: {overall:.2f}% ({n} samples)")

    if task_type_scores:
        eval_logger.info("Task-type Accuracy:")
        for task_type, scores in sorted(task_type_scores.items()):
            avg = sum(scores) / len(scores) * 100.0
            eval_logger.info(f"  [{task_type}]: {avg:.2f}% ({len(scores)} samples)")

    if content_cat_scores:
        eval_logger.info("Content-category Accuracy:")
        for cat, scores in sorted(
            ((k, v) for k, v in content_cat_scores.items() if k is not None),
            key=lambda x: x[0],
        ):
            avg = sum(scores) / len(scores) * 100.0
            eval_logger.info(f"  [{cat}]: {avg:.2f}% ({len(scores)} samples)")

    if duration_scores:
        eval_logger.info("Video-duration Accuracy:")
        for dur, scores in sorted(duration_scores.items()):
            avg = sum(scores) / len(scores) * 100.0
            eval_logger.info(f"  [{dur}]: {avg:.2f}% ({len(scores)} samples)")

    eval_logger.info("=" * 50)
    return overall
