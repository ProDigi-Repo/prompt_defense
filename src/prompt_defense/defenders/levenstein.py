from Levenshtein import ratio


def calculate_levensthein_distance(str1: str, str2: str) -> float:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    return ratio(str1, str2)
