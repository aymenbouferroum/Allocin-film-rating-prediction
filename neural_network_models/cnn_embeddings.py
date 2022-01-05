import os
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import ast
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pnd

tf.config.list_physical_devices("GPU")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
vocab_size = 20000
sequence_length = 1000
embedding_dim = 200


def read_file(link):
    arr = []
    with open(link, "r", encoding="utf-8") as f:
        for line in f:
            arr.append(line)
    return arr


def read_file_notes(link):
    arr2 = []
    with open(link, "r") as f:
        for line in f:
            arr2.append(float(line.replace(",", ".")))
    return arr2


def read_review_id(filename):
    reviewID = []
    for line in open(filename, 'r', encoding='UTF-8'):
        line = ast.literal_eval(line).get("review_id")
        reviewID.append(line)
    return reviewID


if __name__ == '__main__':
    # Load preprocessed data
    x_train = read_file("../data/x_train.txt")[:100000]
    y_train = read_file_notes("../data/y_train.txt")[:100000]
    x_test = read_file("../data/x_test.txt")

    y_train = pnd.get_dummies(y_train)
    y_train = y_train.to_numpy()


    vectorize_layer = TextVectorization(
        standardize=None,
        ngrams=6,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    vectorize_layer.adapt(np.array(x_train))

    # Construction of the CNN model
    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim),

    ])
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))

    # model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    optimiser = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy')

    model.summary()

    csv_logger = CSVLogger('.training.log', append=True)
    checkpoint = ModelCheckpoint(filepath=".best_model.tf", monitor="val_loss", verbose=1,
                                 save_best_only=True, mode="max")

    # Train the model
    model.fit(x_train, y_train, epochs=40, verbose=1, validation_split=0.1, batch_size=128,
              callbacks=[checkpoint, csv_logger])


    # predict the labels on test dataset
    predictions = model.predict(x_test)
    result = open('../exemple_sortie_test.txt', 'w', encoding='utf-8')

    reviewID = read_review_id("../data/test_comments.txt")
    dic = {0: "0,5", 1: "1,0", 2: "1,5", 3: "2,0", 4: "2,5", 5: "3,0", 6: "3,5", 7: "4,0", 8: "4,5", 9: "5,0"}
    for i in range(len(predictions)):
        print(reviewID[i] + " " + dic.get(np.argmax(predictions[i])), file=result)
    result.close()
