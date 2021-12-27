import re

from nltk.stem import PorterStemmer


class Preprocessor:

    STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                  'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                  'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                  'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                  's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o',
                  're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
                  'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
                  'weren', 'won', 'wouldn']

    def __init__(self, text, unwanted_prefixes=None):
        self.words = text.split()
        self.unwanted_prefixes = unwanted_prefixes or ["@", "#", "http", "www"]

    def _to_lower(self):
        self.words = [x.lower() for x in self.words]
        return self

    def _eliminate_unwanted_words(self):
        self.words = list(filter(lambda x: not x.startswith(*self.unwanted_prefixes),
                                 self.words))
        return self

    def _remove_numbers_punctuations(self):
        self.words = [re.sub(r'[^a-z\s]', '', x) for x in self.words]
        return self

    def _remove_stopwords(self):
        self.words = list(filter(lambda x: x not in Preprocessing.STOP_WORDS, self.words))
        return self

    def _stem(self):
        ps = PorterStemmer()
        self.words = [ps.stem(x) for x in self.words]
        return self

    def get_text(self):
        return " ".join([x.strip() for x in self.words if x])

    def preprocess(self, preprocess_steps=None):
        # Order is important
        # Tokenize ->
        # lowercase ->
        # Remove Handles + URLs ->
        # Remove Punctuations + stopwords ->
        # Stem

        if not preprocess_steps:
            preprocess_steps = ["_to_lower", "_eliminate_unwanted_words",
                                "_remove_numbers_punctuations", "_remove_stopwords", "_stem"]

        to_return = self
        for step in preprocess_steps:
            to_return = getattr(to_return, step)()

        return to_return.get_text()
