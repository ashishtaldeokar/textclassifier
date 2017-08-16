"""
Preprocess Module for text to vector format
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class prep():
    """
    Preprocessing of text
    """
    def __init__(self, X_raw, y_raw):
        """
        Initialize the preprocessor
        """
        self.vectorizer = CountVectorizer(ngram_range = (1, 2), min_df = 1)
        self.X_vec = self.vectorizer.fit_transform(X_raw)
        self.transformer = TfidfTransformer()
        self.vector = self.X_vec.toarray()
        self.tfidf = self.transformer.fit_transform(self.vector)
        self.y_enc = LabelEncoder().fit_transform(y_raw)
    def getVector(self):
        """
        return the vector form
        """
        return self.vector
    def getTfidf(self):
        """
        return the Tf-idf form
        """
        return self.tfidf.toarray()
    def get_train_test(self):
        """
        return Tf-idf in train test split
        """
        return train_test_split(self.tfidf, self.y_enc, random_state=14)
