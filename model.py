from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Concatenate, LeakyReLU, Dense
from keras.models import load_model


class GenreClassifier:
    def __init__(self, n_classes, corpus_size, x1_len, x2_len, filename=None):
        if filename is None:
            self.model = self.build_model(n_classes, corpus_size, x1_len, x2_len)
        else:
            self.model = load_model(filename)

    def fit(self, x1_train, x2_train, y_train, epochs=33, batch_size=4048, validation_split=0.2, shuffle=False):
        return self.model.fit([x1_train, x2_train], y_train,
                              validation_split=validation_split,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True)

    def build_model(self, n_classes, corpus_size, x1_len, x2_len):
        input_branch_1 = Input(shape=(x1_len,), dtype='int32')
        embedded_sequences_1 = Embedding(corpus_size, 100)(input_branch_1)

        input_branch_2 = Input(shape=(x2_len,), dtype='int32')
        embedded_sequences_2 = Embedding(corpus_size, 100)(input_branch_2)

        l_lstm_1 = Bidirectional(LSTM(100))(embedded_sequences_1)
        l_lstm_2 = Bidirectional(LSTM(100))(embedded_sequences_2)

        concat = Concatenate()([l_lstm_1, l_lstm_2])

        preds = Dense(n_classes * 4)(concat)
        preds = LeakyReLU(alpha=0.3)(preds)

        preds = Dense(n_classes, activation='sigmoid')(preds)

        model = Model([input_branch_1, input_branch_2], preds)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        return model

    def load(self, filename):
        self.model = load_model(filename)

    def save(self, filename):
        self.model.save(filename)

    def predict(self, in1, in2):
        return self.model.predict([in1, in2])


if __name__ == "__main__":
    test_model = GenreClassifier(1000, "model.pkl")
    print(test_model.predict(
        "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the "
        "scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate "
        "Buzz and Woody from their owner, the duo eventually learns to put aside their differences")
    )
