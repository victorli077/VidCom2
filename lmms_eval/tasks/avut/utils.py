import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

DEFAULT_PRE_PROMPT = ""
DEFAULT_POST_PROMPT = "Respond with only the letter (A, B, C, or D) of the best answer."

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")
CACHE_ROOT = os.path.expanduser(os.getenv("AVUT_CACHE", os.path.join(HF_HOME, "AVUTBenchmark")))


def avut_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter (A/B/C/D) from the doc."""
    return str(doc["answer"]).strip().upper()


def avut_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the textual prompt for audio-centric video understanding.

    The AVUT benchmark focuses on audio content and audio-visual interactions.
    We give a clear instruction to the model to rely on both audio and visual cues.
    """
    pre_prompt = DEFAULT_PRE_PROMPT
    post_prompt = DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    choices = [
        doc.get("option_A", "").strip(),
        doc.get("option_B", "").strip(),
        doc.get("option_C", "").strip(),
        doc.get("option_D", "").strip(),
    ]

    # Format: A. <option_A>\nB. <option_B>\nC. <option_C>\nD. <option_D>
    choice_lines = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]

    prompt_parts = [
        pre_prompt,
        "Select the best answer to the following multiple-choice question based on the video. "
        "You should carefully watch the video and listen to the audio to determine the correct answer. "
        "Respond with only the letter (A, B, C, or D) of the best option.\n",
        doc["question"],
        "\n",
        "\n".join(choice_lines),
        "\n",
        post_prompt,
    ]
    return "".join(prompt_parts).strip()


def avut_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """
    Return the video clip path for the question, resolved from the hub cache.

    The video_path in the JSON is in the format 'data/<video_id>.mp4', but videos
    are stored directly in the HF hub cache snapshot directory (no 'data/' prefix).
    We strip the 'data/' prefix and resolve relative to CACHE_ROOT.
    """
    raw_path = doc.get("video_path", "")
    if not raw_path:
        return []

    # video_path in JSON is "data/xxx.mp4", but cache has "xxx.mp4" directly
    video_name = raw_path.replace("data/", "", 1)
    full_path = os.path.join(CACHE_ROOT, video_name)
    full_path = os.path.expanduser(full_path)
    return [full_path]


def parse_multi_choice_response(response: Optional[str], all_choices: List[str]) -> str:
    """
    Parse the model text output into a choice label.

    Args:
        response: Raw model string output.
        all_choices: Valid labels, e.g. ["A", "B", "C", "D"].

    Returns:
        The parsed label (uppercased). Falls back to the first choice if nothing matches.
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
        "\u261e",  # right pointer
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

    # Fallback: pick the first valid choice to avoid empty return
    return all_choices[0]


def avut_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compare model prediction with ground truth for AVUT benchmark.

    Args:
        doc: Single dataset example (question record from AVUT JSON).
        results: List of model output strings for this example.

    Returns:
        Dict keyed by metric name with per-sample fields used in aggregation.
        Aggregates by task_type to show per-category accuracy.
    """
    pred = results[0] if results else ""
    all_choices = ["A", "B", "C", "D"]
    answer = parse_multi_choice_response(pred, all_choices)
    gt_answer = avut_doc_to_target(doc)
    score = 1.0 if answer == gt_answer else 0.0

    return {
        "avut_score": {
            "question_id": doc.get("QA_id", doc.get("id", "")),
            "task_type": doc.get("task_type", "unknown"),
            "score": score,
        }
    }


def avut_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate per-sample scores into per-task_type accuracy and overall accuracy.

    This is executed in the main process (rank 0) after all samples are processed.

    Args:
        results: List of dicts produced by avut_process_results for the metric.

    Returns:
        Overall accuracy (percentage 0-100).
    """
    task_score_map = defaultdict(list)
    total_score = 0.0

    for result in results:
        task_type = result.get("task_type", "unknown")
        score = float(result.get("score", 0.0))
        task_score_map[task_type].append(score)
        total_score += score

    overall = (total_score / len(results) * 100.0) if results else 0.0

    for task_type, scores in sorted(task_score_map.items()):
        task_avg = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"AVUT ({task_type}): {task_avg:.2f}% ({len(scores)} samples)")

    eval_logger.info(f"Overall AVUT: {overall:.2f}% ({len(results)} samples)")
    return overall
