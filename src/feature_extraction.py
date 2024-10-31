"""
This module handles feature extraction used to determine the casing pattern for each token,
as well as converting tokens to lowercase and writing them to a new file.
"""

from features import extract
from case import get_tc


def feature_extraction():
    """
    Reads a tokenized text file, extracts casing patterns and features for each token,
    and writes the processed data to a new file.
    """

    file_paths = {
        "source": "test_1.tok",
        "replace": "test.tok",
        "lower": "test_lower.tok",
        "test_features": "test_feature_extration",
    }

    with open(file_paths["source"], "r", encoding="utf-8") as source, open(
        file_paths["replace"], "w", encoding="utf-8"
    ) as replace_sink, open(
        file_paths["lower"], "w", encoding="utf-8"
    ) as lower_sink, open(
        file_paths["test_features"], "w", encoding="utf-8"
    ) as features_sink:
        for line in source:
            # replace certain marks
            line = line.replace(":", "_").strip()
            replace_sink.write(line + "\n")

            if line:  # check if the line is not empty
                tokens = line.split()

                # casefold tokens
                token_lower = [token.lower() for token in tokens]
                token_lower_list = "\t".join(token_lower)
                lower_sink.write(token_lower_list + "\n")

                # tag case for each token
                token_tc = [get_tc(token) for token in tokens]

                # extract features for each token
                features = extract(tokens)

                if features is not None:
                    features_sink.write("\n")
                    tab_separate_features = ("\t".join(feature) for feature in features)
                    for tc, tab_feature in zip(token_tc, tab_separate_features):
                        features_sink.write(f"{tc[0]}\t{tab_feature.lower()}\n")
                        # features_sink.write(f"{tab_feature.lower()}\n")
            else:
                lower_sink.write("\n")


if __name__ == "__main__":
    feature_extraction()
