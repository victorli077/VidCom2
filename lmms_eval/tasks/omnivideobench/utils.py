import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger as eval_logger

DEFAULT_PRE_PROMPT = ""
DEFAULT_POST_PROMPT = "Respond with only the letter (A, B, C, or D) of the best answer."

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface")
CACHE_ROOT = os.path.expanduser(os.path.join(HF_HOME, "OmniVideoBench"))


def omnivideobench_doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the ground-truth answer letter (A/B/C/D) from correct_option field."""
    correct = doc.get("correct_option", "")
    if isinstance(correct, str):
        return correct.strip().upper()
    return str(correct).upper()


def omnivideobench_doc_to_text(
    doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build the textual prompt for audio-visual video understanding.

    Dataset fields:
      - question: str
      - options: np.array like ['A.Option1', 'B.Option2', 'C.Option3', 'D.Option4']
      - correct_option: 'A'/'B'/'C'/'D'
    """
    pre_prompt = DEFAULT_PRE_PROMPT
    post_prompt = DEFAULT_POST_PROMPT
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", pre_prompt)
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", post_prompt)

    # options is a numpy array or list of strings like ['A.Option1', 'B.Option2', ...]
    raw_options = doc.get("options", [])
    if isinstance(raw_options, np.ndarray):
        raw_options = raw_options.tolist()

    prompt_parts = [
        pre_prompt,
        "Select the best answer to the following multiple-choice question based on the video. "
        "You should carefully watch the video and listen to the audio to determine the correct answer. "
        "Respond with only the letter (A, B, C, or D) of the best option.\n",
        doc.get("question", ""),
        "\n",
        "\n".join(str(o) for o in raw_options),
        "\n",
        post_prompt,
    ]
    return "".join(prompt_parts).strip()


def omnivideobench_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """
    Return the video path resolved from the hub cache.

    Dataset 'video' field is like 'videos/video_1.mp4'.
    We join it with CACHE_ROOT to get the absolute path.
    """
    video_path = doc.get("video", "")
    if not video_path:
        return []

    full_path = os.path.join(CACHE_ROOT, video_path)
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
        "\u261e",  # right pointer
        "<|im_end|>",
    ]
    for prefix in answer_prefixes:
        response = response.replace(prefix, "")

    response = response.strip()
    response = re.sub(
        r"[.,:!\"'`;\\/?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]", " ", response
    )
    tokens = response.split()

    for token in tokens:
        if token in all_choices or token.upper() in all_choices:
            return token.upper()

    # Fallback: pick the first valid choice to avoid empty return
    return all_choices[0]


def omnivideobench_process_results(
    doc: Dict[str, Any], results: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare model prediction with ground truth for OmniVideoBench benchmark.

    Args:
        doc: Single dataset example. Key fields:
              - correct_option: 'A'/'B'/'C'/'D'
              - question_type: category for per-type accuracy
              - audio_type: audio type for per-type accuracy
        results: List of model output strings for this example.

    Returns:
        Dict keyed by metric name with per-sample fields used in aggregation.
    """
    pred = results[0] if results else ""
    all_choices = ["A", "B", "C", "D"]
    answer = parse_multi_choice_response(pred, all_choices)
    gt_answer = omnivideobench_doc_to_target(doc)
    score = 1.0 if answer == gt_answer else 0.0

    return {
        "omnivideobench_score": {
            "question_id": doc.get("id", ""),
            "question_type": doc.get("question_type", "unknown"),
            "audio_type": doc.get("audio_type", "unknown"),
            "score": score,
        }
    }


def omnivideobench_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """
    Aggregate per-sample scores into per-question-type and per-audio-type accuracy,
    plus overall accuracy.

    This is executed in the main process (rank 0) after all samples are processed.

    Args:
        results: List of dicts produced by omnivideobench_process_results.

    Returns:
        Overall accuracy (percentage 0-100).
    """
    question_type_scores = defaultdict(list)
    audio_type_scores = defaultdict(list)
    total_score = 0.0

    for result in results:
        qt = result.get("question_type", "unknown")
        at = result.get("audio_type", "unknown")
        score = float(result.get("score", 0.0))

        question_type_scores[qt].append(score)
        audio_type_scores[at].append(score)
        total_score += score

    n = len(results)
    overall = (total_score / n * 100.0) if n > 0 else 0.0

    eval_logger.info("=" * 50)
    eval_logger.info(f"Overall OmniVideoBench Accuracy: {overall:.2f}% ({n} samples)")

    if question_type_scores:
        eval_logger.info("Question-type Accuracy:")
        for qt, scores in sorted(question_type_scores.items()):
            avg = sum(scores) / len(scores) * 100.0
            eval_logger.info(f"  [{qt}]: {avg:.2f}% ({len(scores)} samples)")

    if audio_type_scores:
        eval_logger.info("Audio-type Accuracy:")
        for at, scores in sorted(audio_type_scores.items()):
            avg = sum(scores) / len(scores) * 100.0
            eval_logger.info(f"  [{at}]: {avg:.2f}% ({len(scores)} samples)")

    eval_logger.info("=" * 50)
    return overall
