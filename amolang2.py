import numpy as np
import data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

class DenseTransformer():

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

data = data.Data()
allAPI = data.getMalAPI() + data.getNorAPI()
stopWords = data.getStopwords()

Dataset_X = []
Dataset_Y = []

for meta, label in allAPI:
   Dataset_X.append(meta)
   Dataset_Y.append(label)
X_train = np.array(Dataset_X)
Y_train = np.array(Dataset_Y)


# vec = CountVectorizer()
# data = vec.fit_transform(X_train).toarray()

# for i in range (1, 692):
#
#    classifier_SVC = Pipeline([
#       ('vectorizer', CountVectorizer()),
#       ('fs', SelectFromModelExtraTreesClassifier()),
#       ('classifier', SVC(kernel='rbf', C=10000000, gamma=1e-08))
#       ])
#    classifier_LinearSVC = Pipeline([
#       ('vectorizer', CountVectorizer()),
#       ('fs', SelectKBest(chi2, k=i)),
#       ('classifier', LinearSVC())
#       ])
#
#    classifier_NuSVC = Pipeline([
#       ('vectorizer', CountVectorizer()),
#       ('fs', SelectKBest(chi2, k=i)),
#       ('classifier', NuSVC(gamma=0.1, degree=10))
#       ])
#
#    classifier_GaussianNB = Pipeline([
#       ('vectorizer', CountVectorizer()),
#       ('fs', SelectKBest(chi2, k=i)),
#       ('to_dense', DenseTransformer()),
#       ('classifier', BernoulliNB(alpha=0.01, binarize=0.1))
#       ])
#
#    classifier_MultinomialNB = Pipeline([
#       ('vectorizer', CountVectorizer()),
#       ('fs', SelectKBest(chi2, k=i)),
#       ('to_dense', DenseTransformer()),
#       ('classifier', MultinomialNB())
#       ])
#
#    classifier_SVC.fit(X_train, Y_train)
#    classifier_LinearSVC.fit(X_train, Y_train)
#    classifier_NuSVC.fit(X_train, Y_train)
#    classifier_GaussianNB.fit(X_train, Y_train)
#    classifier_MultinomialNB.fit(X_train, Y_train)
#
#    classifier_SVC_scores = cross_validation.cross_val_score(classifier_SVC, X_train, Y_train, cv=10)
#    classifier_LinearSVC_scores = cross_validation.cross_val_score(classifier_LinearSVC, X_train, Y_train, cv=10)
#    classifier_NuSVC_scores = cross_validation.cross_val_score(classifier_NuSVC, X_train, Y_train, cv=10)
#    classifier_GaussianNB_scores = cross_validation.cross_val_score(classifier_GaussianNB, X_train, Y_train, cv=10)
#    classifier_MultinomialNB_scores = cross_validation.cross_val_score(classifier_MultinomialNB, X_train, Y_train, cv=10)
#
#    print  i, ", ",  (classifier_SVC_scores.mean() + classifier_LinearSVC_scores.mean() + classifier_NuSVC_scores.mean() + classifier_GaussianNB_scores.mean() + classifier_MultinomialNB_scores.mean()) / 5
#    # print "Accuracy : %0.2f" % classifier_SVC_scores.mean()
#    # print "Accuracy : %0.2f" % classifier_LinearSVC_scores.mean()
#    # print "Accuracy : %0.2f" % classifier_NuSVC_scores.mean()
#    # print "Accuracy : %0.2f" % classifier_GaussianNB_scores.mean()
#    # print "Accuracy : %0.2f" % classifier_MultinomialNB_scores.mean()

for i in range (1, 692):

   classifier_SVC = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('fs', RFE(LogisticRegression(), i)),
      ('classifier', SVC(kernel='rbf', C=10000000, gamma=1e-08))
      ])
   classifier_LinearSVC = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('fs', RFE(LogisticRegression(), i)),
      ('classifier', LinearSVC())
      ])

   classifier_NuSVC = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('fs', RFE(LogisticRegression(), i)),
      ('classifier', NuSVC(gamma=0.1, degree=10))
      ])

   classifier_GaussianNB = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('fs', RFE(LogisticRegression(), i)),
      ('to_dense', DenseTransformer()),
      ('classifier', BernoulliNB(alpha=0.01, binarize=0.1))
      ])

   classifier_MultinomialNB = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('fs', RFE(LogisticRegression(), i)),
      ('to_dense', DenseTransformer()),
      ('classifier', MultinomialNB())
      ])

   classifier_SVC.fit(X_train, Y_train)
   classifier_LinearSVC.fit(X_train, Y_train)
   classifier_NuSVC.fit(X_train, Y_train)
   classifier_GaussianNB.fit(X_train, Y_train)
   classifier_MultinomialNB.fit(X_train, Y_train)

   classifier_SVC_scores = cross_validation.cross_val_score(classifier_SVC, X_train, Y_train, cv=10)
   classifier_LinearSVC_scores = cross_validation.cross_val_score(classifier_LinearSVC, X_train, Y_train, cv=10)
   classifier_NuSVC_scores = cross_validation.cross_val_score(classifier_NuSVC, X_train, Y_train, cv=10)
   classifier_GaussianNB_scores = cross_validation.cross_val_score(classifier_GaussianNB, X_train, Y_train, cv=10)
   classifier_MultinomialNB_scores = cross_validation.cross_val_score(classifier_MultinomialNB, X_train, Y_train, cv=10)

   print  i, ", ",  (classifier_SVC_scores.mean() + classifier_LinearSVC_scores.mean() + classifier_NuSVC_scores.mean() + classifier_GaussianNB_scores.mean() + classifier_MultinomialNB_scores.mean()) / 5
   # print "Accuracy : %0.2f" % classifier_SVC_scores.mean()
   # print "Accuracy : %0.2f" % classifier_LinearSVC_scores.mean()
   # print "Accuracy : %0.2f" % classifier_NuSVC_scores.mean()
   # print "Accuracy : %0.2f" % classifier_GaussianNB_scores.mean()
   # print "Accuracy : %0.2f" % classifier_MultinomialNB_scores.mean()
