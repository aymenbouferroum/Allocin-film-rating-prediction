from sklearnex import patch_sklearn
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import ast
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from zeugma.embeddings import EmbeddingTransformer

patch_sklearn()
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
    x_train = read_file("../data/x_train.txt")
    y_train = read_file_notes("../data/y_train.txt")
    x_test = read_file("../data/x_test.txt")

    # Encode target labels with values between 0 and 9
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)

    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    vectorize_layer.adapt(np.array(x_train))

    # Use pretrained embedding model Glove
    glove = EmbeddingTransformer("glove")
    x_train = glove.transform(x_train)
    x_test = glove.transform(x_test)

    # Apply Linear Support Vector Machine model
    SVM = svm.SVC(C=0.68, kernel='linear', gamma='auto')
    SVM.fit(x_train, y_train)

    # predict the labels on test dataset
    predictions_SVM = SVM.predict(x_test)
    result = open('../exemple_sortie_test.txt', 'w', encoding='utf-8')

    reviewID = read_review_id("../data/test_comments.txt")
    dic = {0: "0,5", 1: "1,0", 2: "1,5", 3: "2,0", 4: "2,5", 5: "3,0", 6: "3,5", 7: "4,0", 8: "4,5", 9: "5,0"}
    for i in range(len(predictions_SVM)):
        print(reviewID[i] + " " + dic.get(predictions_SVM[i]), file=result)
