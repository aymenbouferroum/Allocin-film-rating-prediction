import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pnd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def read_file(link):
    arr = []
    with open(link, "r", encoding="utf-8") as f:
        for line in f:
            arr.append(line)
    return arr


def read_review_id(filename):
    reviewID = []
    for line in open(filename, 'r', encoding='UTF-8'):
        line = ast.literal_eval(line).get("review_id")
        reviewID.append(line)
    return reviewID


def read_file_notes(link):
    arr2 = []
    with open(link, "r") as f:
        for line in f:
            arr2.append(float(line.replace(",", ".")))
    return arr2


if __name__ == '__main__':
    # Load preprocessed data
    x_train = read_file("../data/x_train.txt")[:100000]
    y_train = read_file_notes("../data/y_train.txt")[:100000]
    x_test = read_file("../data/x_test.txt")

    y_train = pnd.get_dummies(y_train)
    y_train = y_train.to_numpy()

    # Convert a collection of raw documents to a matrix of  TF-IDF features
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    # Construction of the MLP model
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimiser = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()
    # Train the model
    model.fit(x_train, y_train, epochs=20, verbose=1, batch_size=64, validation_split=0.1)


    # predict the labels on test dataset
    predictions = model.predict(x_test)
    result = open('../exemple_sortie_test.txt', 'w', encoding='utf-8')

    reviewID = read_review_id("../data/test_comments.txt")
    dic = {0: "0,5", 1: "1,0", 2: "1,5", 3: "2,0", 4: "2,5", 5: "3,0", 6: "3,5", 7: "4,0", 8: "4,5", 9: "5,0"}
    for i in range(len(predictions)):
        print(reviewID[i] + " " + dic.get(np.argmax(predictions[i])), file=result)

    result.close()
