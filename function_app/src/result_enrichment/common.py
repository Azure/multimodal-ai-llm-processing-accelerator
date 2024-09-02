from typing import Callable, Iterable, Optional


def is_exact_match(value: str, content: str) -> bool:
    """
    Check if a value is an exact match for a piece of content.

    :param value:
        The value to check.
    :param content:
        The content to check against.

    :return:
        True if the value is an exact match for the content, False otherwise.
    """
    return value == content


def is_value_in_content(value: str, content: str) -> bool:
    """
    Check if a value is contained with a piece of content.

    :param value:
        The value to check.
    :param content:
        The content to check against.

    :return:
        True if the value is contained within the content, False otherwise.
    """
    return value in content


def merge_confidence_scores(
    scores: Iterable[float],
    no_values_replacement: Optional[float | int] = None,
    multiple_values_replacement_func: Callable = min,
) -> float:
    """
    Merge multiple confidence scores into a single score.

    :param scores:
        An iterable of confidence scores to merge.
    :param no_values_replacement:
        The value to return if no confidence scores are provided.
    :param multiple_values_replacement_func:
        The function to use to merge multiple confidence scores. Default is
        `min`.

    :return:
        The merged confidence score.
    """
    if len(scores) == 1:
        return scores[0]
    if len(scores) == 0:
        return no_values_replacement
    return multiple_values_replacement_func(scores)
