"""
This module evaluates the accuracy of predicted tokens against gold-standard tokens.
"""

import argparse
import logging


def read_data(file_path: str):
    """Reads and splits data from a file into a list of lists.

    Args:
        file_path: Path to the input file.

    Returns:
        A list of token lists from the input file.
    """

    data_list = []
    with open(file_path, "r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                data_list.append(line.split(" "))
    return data_list


def evaluation(gold_file: str, pred_file: str):
    """Evaluates the accuracy of predicted tokens compared to gold-standard tokens.
    Args:
        gold_file: Path to the gold-standard file.
        pred_file: Path to the predicted tokens file.
    """

    gold = read_data(gold_file)
    pred = read_data(pred_file)
    correct = []
    size = []

    for gold_line, pred_line in zip(gold, pred):
        gold_tokens = [line.split(" ") for line in gold_line]
        pred_tokens = [line.split(" ") for line in pred_line]
        for gold_token, pred_token in zip(gold_tokens, pred_tokens):
            if gold_token == pred_token:
                correct.append(pred_token)
            size.append(gold_token)

    rate = len(correct) / len(size)
    logging.info("Accuracy: %.3f", rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold_file", required=True, help="Path to the gold file.")
    parser.add_argument("--pred_file", required=True, help="Path to prediction file")
    args = parser.parse_args()
    evaluation(args.gold_file, args.pred_file)
