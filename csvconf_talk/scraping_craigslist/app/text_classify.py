from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


#http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def train_classifier(text,labels):
    #http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#parameter-tuning-using-grid-search
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__kernel': ('rbf', 'linear'),
                  'clf__gamma': (1e-3, 1e-4),
                  'clf__C': (1, 10, 100, 1000),
                  }

    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SVC()),])

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    return gs_clf.fit(text,labels)
        
def classify_text(classifier,input_data):
    return classifier.predict([input_data])[0]

