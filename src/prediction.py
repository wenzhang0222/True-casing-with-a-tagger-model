"""
This script uses a CRF model for tagging tokens and applies mixed-case transformations.
"""

import argparse
import re
import json
from typing import Pattern
from collections import defaultdict, Counter
from enum import Enum

import pycrfsuite  # type: ignore


# Function to handle command-line arguments using argparse
def parse_args():
    """Parse command-line arguments for file paths and model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crf_model", required=True, help="Path to the CRF model.")
    parser.add_argument(
        "--test_features", required=True, help="Path to the test features file."
    )
    parser.add_argument(
        "--predictions", required=True, help="Path to the predictions output file."
    )
    parser.add_argument("--token_file", required=True, help="Path to the token file.")
    parser.add_argument(
        "--json_file", required=True, help="Path to the JSON file for storing patterns."
    )
    parser.add_argument(
        "--output", required=True, help="Path to the final output file."
    )
    parser.add_argument(
        "--test_tok",
        required=True,
        help="Path to the test token file for matching patterns.",
    )

    return parser.parse_args()


def read_features(filename: str) -> list:
    """
    Reads features from a file and stores them in a list of a list for each line.
    Inserts blank lines between sentences.

    Args:
        filename (str): Path to the file containing features.

    Returns:
        list: A list containing features for each line.
    """

    list_features = []
    with open(filename, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:  # check line is not empty
                features = [line.split("\t")]
                list_features.append(features)
            else:  # write a blink line between each sentence
                list_features.append("")
    return list_features


def make_predictions(args):
    """
    Makes predictions based on extracted features and writes the results to a file.
    Args:
        args: Parsed command-line arguments.
    """

    tagger = pycrfsuite.Tagger()  # Load the CRF model
    tagger.open(args.crf_model)

    list_list_features = read_features(args.test_features)

    with open(args.predictions, "w", encoding="utf-8") as sink:
        for lists in list_list_features:
            predictions = tagger.tag(lists)
            if predictions:
                for prediction in predictions:
                    sink.write(prediction + "\t")
            else:
                sink.write("\n")


class TokenCase(Enum):
    """Class representing different token casing patterns."""

    DC = "DC"
    LOWER = "LOWER"
    UPPER = "UPPER"
    TITLE = "TITLE"
    MIXED = "MIXED"


class UnknownTokenCaseError(ValueError):
    """Custom error to handle unknown token cases."""

    pass


def apply_cc(char: str, case: str) -> str:
    """Apply character case."""

    if case == "UPPER":
        return char.upper()
    elif case == "LOWER":
        return char.lower()
    # Add other cases as needed
    return char


def apply_tc(nunistr: str, tc: TokenCase, pattern: Pattern = None) -> str:
    """Applies TokenCase to a Unicode string.

    This function applies a TokenCase to a Unicode string. Unless TokenCase is
    `DC`, this is insensitive to the casing of the input string.

    Args:
        nunistr: A Unicode string to be cased.
        tc: A TokenCase indicating the casing to be applied.
        pattern: An iterable of CharCase characters representing the specifics
            of the `MIXED` TokenCase, when the `tc` argument is `MIXED`.

    Returns:
        An appropriately-cased Unicode string.

    Raises:
        UnknownTokenCaseError.
    """

    if tc == TokenCase.DC:
        return nunistr
    elif tc == TokenCase.LOWER:
        return nunistr.lower()
    elif tc == TokenCase.UPPER:
        return nunistr.upper()
    elif tc == TokenCase.TITLE:
        return nunistr.title()
    elif tc == TokenCase.MIXED:
        # Defaults to lowercase if no pattern is provided.
        if pattern is None:
            return nunistr.lower()
        assert pattern
        assert len(nunistr) == len(pattern)
        return "".join(apply_cc(ch, cc) for (ch, cc) in zip(nunistr, pattern))
    raise UnknownTokenCaseError(tc)


def read_tokens(file_path: str):
    """Reads tokens from a file."""

    list_tokens = []
    with open(file_path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                list_tokens.append(line.split("\t"))
    return list_tokens


def read_tags(file_path: str):
    """Reads tags from a file."""

    list_tags = []
    with open(file_path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                list_tags.append(line.split("\t"))
    return list_tags


def count_patterns(args):
    """Counts mixed-case patterns in tokens.
    Args:
        args: Parsed command-line arguments."""

    mcdict = defaultdict(Counter)
    pattern_re = re.compile(r"[a-z]*[A-Z][a-z]*")

    with open(args.test_tok, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                _tokens = line.split()
                for token in _tokens:
                    mix_patterns = pattern_re.findall(token)
                    if mix_patterns:
                        for pattern in mix_patterns:
                            mcdict[token.casefold()][pattern] += 1

    token_to_most_frequent_pattern = {}
    for token, patterns_counter in mcdict.items():
        most_frequent_pattern = patterns_counter.most_common(1)
        if most_frequent_pattern:
            token_to_most_frequent_pattern[token] = most_frequent_pattern[0][0]

    # Write the token pattern dictionary to a JSON file
    with open(args.json_file, "w", encoding="utf-8") as json_fp:
        json.dump(token_to_most_frequent_pattern, json_fp)


def apply_tag_token(args, all_tokens, all_tags, _mixed_case_dict):
    """Applies tags to tokens and writes the output to a file.
    Args:
        args: Parsed command-line arguments.
        all_tokens: List of tokens.
        all_tags: List of corresponding tags.
        _mixed_case_dict: A dictionary of mixed-case patterns."""

    with open(args.output, "w", encoding="utf-8") as sink:
        assert len(all_tokens) == len(all_tags)
        for token_list, tag_list in zip(all_tokens, all_tags):
            result = []
            for token, tag in zip(token_list, tag_list):
                try:
                    # Convert the tag (string) to a TokenCase enum
                    tag_enum = TokenCase[tag]  # Convert string tag to TokenCase
                except KeyError:
                    raise UnknownTokenCaseError(f"Unknown tag: {tag}")

                if tag_enum == TokenCase.MIXED:
                    pattern = _mixed_case_dict.get(token.casefold())
                    if pattern is None or len(token) != len(pattern):
                        continue
                    tagged_token = apply_tc(token, tag_enum, pattern)
                else:
                    tagged_token = apply_tc(token, tag_enum)
                result.append(tagged_token)
            sink.write(" ".join(result) + "\n")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Make predictions and save them to the predictions file
    make_predictions(args)

    # Count mixed-case patterns and save them to the JSON file
    count_patterns(args)

    # Read tokens and tags from respective files
    tokens = read_tokens(args.token_file)
    tags = read_tags(args.predictions)

    # Load mixed-case dictionary from the JSON file
    with open(args.json_file, "r", encoding="utf-8") as json_file:
        mixed_case_dict = json.load(json_file)

    # Apply tags to tokens and save the output
    apply_tag_token(args, tokens, tags, mixed_case_dict)
