import ast
from scipy import sparse
from sklearnex import patch_sklearn
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
patch_sklearn()

"""
    A function to read polarity file
"""
def read_polarity_file(filename):
    dict_pos = {}
    dict_neg = {}
    dict_neut = {}
    for line in open(filename, 'r', encoding='Latin-1'):
        store = []
        line = line.replace("\n", "").split('"')
        store.append(line[1])
        store.extend(line[2].split(";")[1:])
        line = store
        if int(line[1]) > int(line[2]) and int(line[1]) > int(line[3]):
            dict_pos[line[0]] = int(line[1])
        elif int(line[2]) >= int(line[1]) and int(line[2]) >= int(line[3]):
            dict_neut[line[0]] = int(line[2])
        elif int(line[3]) > int(line[1]) and int(line[3]) > int(line[2]):
            dict_neg[line[0]] = int(line[3])
        else:
            dict_neut[line[0]] = int(line[2])

    return dict_pos, dict_neg, dict_neut

"""
    A function to calculate the polarity of each document
"""
def calculate_polarity(x_train, dict_pos, dict_neg, dict_neut):
    count_pos_list = []
    count_neg_list = []
    count_neut_list = []

    for i in range(len(x_train)):
        count_pos = 0
        count_neg = 0
        count_neut = 0
        res = x_train[i].split()
        for w in res:
            if w in dict_pos.keys():
                count_pos += dict_pos[w] / 50000
            elif w in dict_neg.keys():
                count_neg += dict_neg[w] / 50000
            elif w in dict_neut.keys():
                count_neut += dict_neut[w] / 50000
            else:
                pass
        count_pos_list.append(count_pos)
        count_neg_list.append(count_neg)
        count_neut_list.append(count_neut)
        print(i, " of ", len(x_train))
    return count_pos_list, count_neg_list, count_neut_list


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
    x_train = read_file("../data/x_train.txt")
    y_train = read_file_notes("../data/y_train.txt")
    x_test = read_file("../data/x_test.txt")

    print(len(x_train))
    print(len(y_train))


    dict_pos = {}
    dict_neg = {}
    dict_neut = {}
    dict_pos, dict_neg, dict_neut = read_polarity_file("../data/polarity_fr.txt")

    count_pos_list, count_neg_list, count_neut_list = calculate_polarity(x_train, dict_pos, dict_neg, dict_neut)
    count_pos_list_test, count_neg_list_test, count_neut_list_test = calculate_polarity(x_test, dict_pos, dict_neg,
                                                                                        dict_neut)
    count_pos_list = np.array(count_pos_list).reshape((len(count_pos_list), 1))
    count_neg_list = np.array(count_neg_list).reshape((len(count_neg_list), 1))
    count_neut_list = np.array(count_neut_list).reshape((len(count_neut_list), 1))
    count_pos_list_test = np.array(count_pos_list_test).reshape((len(count_pos_list_test), 1))
    count_neg_list_test = np.array(count_neg_list_test).reshape((len(count_neg_list_test), 1))
    count_neut_list_test = np.array(count_neut_list_test).reshape((len(count_neut_list_test), 1))

    # Encode target labels with values between 0 and 9
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)

    # Convert a collection of raw documents to a matrix of  TF-IDF features
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

    vectorizer.fit(x_train)
    tfidf_train = vectorizer.transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    tfidf_train = tfidf_train.toarray()
    tfidf_test = tfidf_test.toarray()


    x_train = np.concatenate((tfidf_train, count_pos_list, count_neut_list, count_neg_list), axis=1)
    x_test = np.concatenate((tfidf_test, count_pos_list_test, count_neut_list_test, count_neg_list_test), axis=1)

    x_train = sparse.csr_matrix(x_train)
    x_test = sparse.csr_matrix(x_test)


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
