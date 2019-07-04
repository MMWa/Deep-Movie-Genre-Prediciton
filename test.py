from joblib import load
import numpy as np
from model import GenreClassifier, preprocess_input
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True, type=str)
    parser.add_argument("--description", required=True, type=str)

    parser.add_argument("--title_max_length", required=False, type=int, default=20)
    parser.add_argument("--description_max_length", required=False, type=int, default=185)

    parser.add_argument("--labeler_file", required=False, type=str, default="model/class_labeler.joblib")
    parser.add_argument("--tokenizer_file", required=False, type=str, default="model/tokenizer.joblib")

    parser.add_argument("--model_file", required=False, type=str, default="model/weights-epoch-32-val-acc-0.420.h5")

    args = parser.parse_args()

    # load the labeler and tokenizer which contain the genres and word tokens used by the model.
    # we reuse the tokenizer from training, so the input will have the same int token when fed to the neural network.
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

    # predicting the class of the input
    prediction = model.predict(in_title, in_description)
    # since there is a recurrent aspect to out model an input can have multiple outputs,
    # the simplest thing to do is take the mean of all predictions, this can be improved....
    combined_probabilities = np.mean(prediction, axis=0)

    # argmax to find which class has the highest probability
    idx = np.argmax(combined_probabilities)
    # get the name of that class using the index from the labeler
    predicted_genre = labeler.classes_[idx]

    # with python we can only print and not return when at entry point
    print({"title": args.title, "description": args.description, "genre": predicted_genre})
