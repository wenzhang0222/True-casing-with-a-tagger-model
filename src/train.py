"""
This module trains a CRF model using pycrfsuite and saves it as 'model.crfsuite'.
It reads training data from a feature extraction file, processes it, and
then trains a CRF model with the specified parameters.
"""

import argparse
import pycrfsuite  # type: ignore


def read_data(filename: str):
    """
    Reads the data from a file and splits it into features (X) and labels (Y).
    Args:
        filename (str): Path to the file.
    Returns:
        tuple: A tuple containing two lists, X (features) and Y (labels).
    """

    features, labels = [], []

    with open(filename, "r", encoding="utf-8") as source:
        for line in source:
            if line.strip():  # check if the line is not empty
                tokens = line.split("\t")
                label = [tokens[0]]
                feature = [tokens[1:]]
                features.append(feature)
                labels.append(label)
            else:
                features.append("")
                labels.append("")
    return features, labels


def train_model(args: argparse.Namespace):
    """
    Trains a CRF model using pycrfsuite and saves the model as 'model.crfsuite'.
    """

    trainer = pycrfsuite.Trainer(verbose=False)
    features_train, labels_train = read_data(args.train_features)

    for xseq, yseq in zip(features_train, labels_train):
        if len(features_train) == len(labels_train):
            trainer.append(xseq, yseq)

    trainer.set_params(
        {
            "c1": 1.0,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 50,  # stop earlier
            # include transitions that are possible, but not observed
            "feature.possible_transitions": True,
        }
    )

    trainer.train(args.crf_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crf_model", required=True, help="Path to the CRF model.")
    parser.add_argument(
        "--train_features", required=True, help="Path to train features file"
    )
    train_model(parser.parse_args())
