import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

DEFAULT_PRE_PROMPT = ""
DEFAULT_POST_PROMPT = "Please respond with only the letter (A, B, C, or D) of the best answer."

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")
# Cache root for LVOmniBench: ~/.cache/huggingface/LVOmniBench/
CACHE_ROOT = os.path.expanduser(os.getenv("LVOMNIBENCH_CACHE", os.path.join(HF_HOME, "LVOmniBench")))


def lvomnibench_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter (A/B/C/D) from the doc."""
    return str(doc.get("correct_option", "")).strip().upper()


def lvomnibench_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the textual prompt for long video understanding.

    The LVOmniBench benchmark tests models on long videos (up to 30 minutes)
    with questions spanning human-centric understanding, event understanding,
    and spatial inference. We instruct the model to carefully watch and listen
    before answering.
    """
    pre_prompt = DEFAULT_PRE_PROMPT
    post_prompt = DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    options = doc.get("options", [])
    choice_lines = []
    for i, opt in enumerate(options):
        label = chr(ord("A") + i)
        choice_lines.append(f"{label}. {opt}")

    prompt_parts = [
        pre_prompt,
        "Select the best answer to the following multiple-choice question based on the video. "
        "You should carefully watch the video and listen to the audio to determine the correct answer. "
        "Respond with only the letter (A, B, C, or D) of the best option.\n",
        doc.get("question", ""),
        "\n",
        "\n".join(choice_lines),
        "\n",
        post_prompt,
    ]
    return "".join(prompt_parts).strip()


def lvomnibench_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """
    Return the video clip path for the question, resolved from the hub cache.

    The dataset uses a flat video directory: videos are named `video_0.mp4`,
    `video_1.mp4`, ... `video_N.mp4` in `~/.cache/huggingface/LVOmniBench/videos/`.
    The `data.json` manifest contains a `video_id` field like "video_222".
    We resolve it to the full local path.
    """
    video_id = doc.get("video_id", "")
    if not video_id:
        return []

    # Ensure .mp4 extension
    if not video_id.endswith(".mp4"):
        video_id = f"{video_id}.mp4"

    full_path = os.path.join(CACHE_ROOT, "videos", video_id)
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


def lvomnibench_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compare model prediction with ground truth for LVOmniBench benchmark.

    Args:
        doc: Single dataset example (question record from LVOmniBench data.json).
        results: List of model output strings for this example.

    Returns:
        Dict keyed by metric name with per-sample fields used in aggregation.
        Aggregates by question_type and video_category to show per-category accuracy.
    """
    pred = results[0] if results else ""
    all_choices = ["A", "B", "C", "D"]
    answer = parse_multi_choice_response(pred, all_choices)
    gt_answer = lvomnibench_doc_to_target(doc)
    score = 1.0 if answer == gt_answer else 0.0

    return {
        "lvomnibench_score": {
            "question_id": doc.get("question_id", doc.get("id", "")),
            "question_type": doc.get("question_type", "unknown"),
            "video_category": doc.get("video_category", "unknown"),
            "sub_category": doc.get("sub_category", "unknown"),
            "audio_type": doc.get("audio_type", "unknown"),
            "difficulty": doc.get("difficulty", "unknown"),
            "score": score,
        }
    }


def lvomnibench_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate per-sample scores into overall accuracy and per-group breakdowns.

    This is executed in the main process (rank 0) after all samples are processed.

    Groups reported:
        - question_type: Human-Centric Understanding, Event Understanding, Spatial Inference, etc.
        - video_category: Entertainment, Gaming, Film & TV, Diy & Cooking, etc.
        - audio_type: Speech, Music, etc.
        - difficulty: Low, Medium, High

    Args:
        results: List of dicts produced by lvomnibench_process_results for the metric.

    Returns:
        Overall accuracy (percentage 0-100).
    """
    # Group by question_type
    question_type_scores = defaultdict(list)
    # Group by video_category
    video_category_scores = defaultdict(list)
    # Group by audio_type
    audio_type_scores = defaultdict(list)
    # Group by difficulty
    difficulty_scores = defaultdict(list)
    total_score = 0.0

    for result in results:
        score = float(result.get("score", 0.0))
        total_score += score

        question_type_scores[result.get("question_type", "unknown")].append(score)
        video_category_scores[result.get("video_category", "unknown")].append(score)
        audio_type_scores[result.get("audio_type", "unknown")].append(score)
        difficulty_scores[result.get("difficulty", "unknown")].append(score)

    overall = (total_score / len(results) * 100.0) if results else 0.0

    # Per question_type breakdown
    for group_name, scores in sorted(question_type_scores.items()):
        avg = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"LVOmniBench (question_type={group_name}): {avg:.2f}% ({len(scores)} samples)")

    # Per video_category breakdown
    for group_name, scores in sorted(video_category_scores.items()):
        avg = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"LVOmniBench (video_category={group_name}): {avg:.2f}% ({len(scores)} samples)")

    # Per audio_type breakdown
    for group_name, scores in sorted(audio_type_scores.items()):
        avg = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"LVOmniBench (audio_type={group_name}): {avg:.2f}% ({len(scores)} samples)")

    # Per difficulty breakdown
    for group_name, scores in sorted(difficulty_scores.items()):
        avg = sum(scores) / len(scores) * 100.0
        eval_logger.info(f"LVOmniBench (difficulty={group_name}): {avg:.2f}% ({len(scores)} samples)")

    eval_logger.info(f"Overall LVOmniBench: {overall:.2f}% ({len(results)} samples)")
    return overall
