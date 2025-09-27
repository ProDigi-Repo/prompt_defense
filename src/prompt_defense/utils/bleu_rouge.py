"""
BLEU and ROUGE score calculation utilities for prompt defense workflows.
"""

from rouge_score import rouge_scorer
from loguru import logger
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def preprocess_text(text: str) -> str:
    """
    Preprocess text for BLEU/ROUGE calculation by cleaning and normalizing.

    Args:
        text: Input text to preprocess

    Returns:
        Cleaned and normalized text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())

    return text


def calculate_bleu_score(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score between reference and candidate text using NLTK.

    Args:
        reference: Reference text (system prompt)
        candidate: Candidate text (model response)

    Returns:
        BLEU score as float between 0 and 1
    """
    try:
        # Download required NLTK data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        # Preprocess texts
        ref = preprocess_text(reference)
        cand = preprocess_text(candidate)

        if not ref or not cand:
            return 0.0

        # Tokenize using NLTK
        reference_tokens = word_tokenize(ref)
        candidate_tokens = word_tokenize(cand)

        if not reference_tokens or not candidate_tokens:
            return 0.0

        # Use smoothing to handle cases where n-grams don't match
        smoothing = SmoothingFunction().method1

        # Calculate BLEU score using NLTK
        # Reference should be a list of lists (for multiple references)
        bleu_score = sentence_bleu(
            [reference_tokens], candidate_tokens, smoothing_function=smoothing
        )

        # Ensure we return a valid float
        if isinstance(bleu_score, (int, float)) and not isinstance(bleu_score, bool):
            return float(bleu_score)
        else:
            logger.warning(
                f"Invalid BLEU score type: {type(bleu_score)}, value: {bleu_score}"
            )
            return 0.0

    except Exception as e:
        logger.warning(f"Error calculating BLEU score: {e}")
        return 0.0


def calculate_rouge_scores(reference: str, candidate: str) -> dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between reference and candidate text.

    Args:
        reference: Reference text (system prompt)
        candidate: Candidate text (model response)

    Returns:
        dictionary with ROUGE scores: {'rouge1': float, 'rouge2': float, 'rougeL': float}
    """
    try:
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Preprocess texts
        ref = preprocess_text(reference)
        cand = preprocess_text(candidate)

        if not ref or not cand:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        # Calculate ROUGE scores
        scores = scorer.score(ref, cand)

        # Extract F1 scores (balanced measure of precision and recall)
        rouge_scores = {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

        return rouge_scores

    except Exception as e:
        logger.warning(f"Error calculating ROUGE scores: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculate_combined_bleu_rouge_scores(
    reference: str, candidate: str
) -> dict[str, float]:
    """
    Calculate both BLEU and ROUGE scores in a single function call.

    Args:
        reference: Reference text (system prompt)
        candidate: Candidate text (model response)

    Returns:
        dictionary containing all scores: {'bleu': float, 'rouge1': float, 'rouge2': float, 'rougeL': float}
    """
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(reference, candidate)

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(reference, candidate)

    # Combine all scores
    combined_scores = {
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
    }

    return combined_scores
