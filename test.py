from joblib import load
import numpy as np
from model import GenreClassifier, preprocess_input
import sys
import argparse
from keras.preprocessing.sequence import pad_sequences


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True, type=str)
    parser.add_argument("--description", required=True, type=str)

    parser.add_argument("--title_max_length", required=False, type=int, default=20)
    parser.add_argument("--description_max_length", required=False, type=int, default=184)

    parser.add_argument("--labeler_file", required=False, type=str, default="model/class_labler.joblib")
    parser.add_argument("--tokenizer_file", required=False, type=str, default="model/tokenizer.joblib")

    parser.add_argument("--model_file", required=False, type=str, default="model/simple_text_classifier_33.h5")

    args = parser.parse_args()

    labeler = load(args.labeler_file)
    tokenizer = load(args.tokenizer_file)

    num_classes = 32
    MAX_NB_WORDS = 50000
    X1_max_len = args.title_max_length
    X2_max_len = args.description_max_length

    in_title = preprocess_input(tokenizer, args.title, X1_max_len)
    in_description = preprocess_input(tokenizer, args.description, X2_max_len)

    model = GenreClassifier(num_classes,
                            MAX_NB_WORDS,
                            X1_max_len,
                            X2_max_len,
                            filename=args.model_file)

    prediction = model.predict(in_title, in_description)
    combined_probabilities = np.mean(prediction, axis=0)

    idx = np.argmax(combined_probabilities)
    predicted_genre = labeler.classes_[idx]

    print({"title": args.title, "description": args.description, "genre": predicted_genre})
