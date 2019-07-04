from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Concatenate, LeakyReLU, Dense, SpatialDropout1D, \
    BatchNormalization, Dropout
from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences


def preprocess_input(tokenizer, txt_in, len_constant):
    x = tokenizer.texts_to_sequences(texts=txt_in)
    return pad_sequences(x, maxlen=len_constant)


"""
this class creates the model used in training/ prediction
it allows us to either create a model using the build model function,
or load a model graph if the file name is included
n_classes - is the number of classes in the output
corpus_size - refers to the number of tokenized words
x1_len - the padded sequence size of any title
x2_len - the padded sequence size  of any description
filename - a pre-trained model in the form of .h5 can be used in inference instances
"""


class GenreClassifier:
    def __init__(self, n_classes=None, corpus_size=None, x1_len=None, x2_len=None, filename=None):
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
        embedded_sequences_1 = SpatialDropout1D(0.2)(embedded_sequences_1)

        input_branch_2 = Input(shape=(x2_len,), dtype='int32')
        embedded_sequences_2 = Embedding(corpus_size, 100)(input_branch_2)
        embedded_sequences_2 = SpatialDropout1D(0.2)(embedded_sequences_2)

        l_lstm_1 = Bidirectional(LSTM(200))(embedded_sequences_1)
        l_lstm_2 = Bidirectional(LSTM(200))(embedded_sequences_2)

        concat = Concatenate()([l_lstm_1, l_lstm_2])
        concat = BatchNormalization()(concat)

        preds = Dense(n_classes * 8)(concat)
        preds = LeakyReLU(alpha=0.3)(preds)
        preds = Dropout(0.15)(preds)

        preds = Dense(n_classes, activation='sigmoid')(preds)

        model = Model([input_branch_1, input_branch_2], preds)
        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['acc'])
        return model

    def load(self, filename):
        self.model = load_model(filename)

    def save(self, filename):
        self.model.save(filename)

    def predict(self, in1, in2):
        return self.model.predict([in1, in2])
