from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearnex import patch_sklearn
import ast
from sklearn.preprocessing import LabelEncoder

patch_sklearn()


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

    # Convert a collection of raw documents to a matrix of  TF-IDF features
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    # Apply logistic regression model
    model = LogisticRegression(verbose=1, C=0.9, penalty='l2', max_iter=1000, random_state=0)
    model.fit(x_train, y_train)

    """
    grid search 
    """
    # model = LogisticRegression(max_iter=100, verbose=1)
    # solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga', 'sag']
    # penalty = ['l2']
    # c_values = [ 1.0, 0.7, 2.0, 0.8 ]
    # grid = dict(solver=solvers, penalty=penalty, C=c_values)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0, verbose=1)
    # grid_result = grid_search.fit(tfidf_train, Train_Y)

    # predict the labels on test dataset
    predictions_SVM = model.predict(x_test)
    result = open('../output/exemple_sortie_test.txt', 'w', encoding='utf-8')

    reviewID = read_review_id("../data/test_comments.txt")
    dic = {0: "0,5", 1: "1,0", 2: "1,5", 3: "2,0", 4: "2,5", 5: "3,0", 6: "3,5", 7: "4,0", 8: "4,5", 9: "5,0"}
    for i in range(len(predictions_SVM)):
        print(reviewID[i] + " " + dic.get(predictions_SVM[i]), file=result)
