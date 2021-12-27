import pickle as pkl


class FeatureEngineer:
    def __init__(self, labelled_corpus, feature_method="freq_dict"):
        # Of format: Pandas Dataframe : Text : Label
        self.feature_method = "_"+feature_method
        self.features = getattr(FeatureEngineer, self.feature_method)(labelled_corpus.fillna(""))

    @staticmethod
    def _freq_dict(corpus_df):
        feature_dict = {}

        for item in corpus_df.iteritems():
            class_label = item["Label"]

            for word in item["Text"].split():
                if word not in feature_dict:
                    feature_dict[word] = {}
                feature_dict[word][class_label] = feature_dict[word].get(class_label, 0) + 1

        return feature_dict

    def _freq_dict_vectorize(self, text):
        words = text.split()
        to_return = {}

        for word in words:
            val_lkpup = self.features.get(word, {})
            for key, val in val_lkpup.items():
                to_return[key] = to_return.get(key, 0) + val

        to_return_vector = [1]
        for key

    def get_features(self):
        return self.features

    def save_features(self, save_path="features.pkl"):
        with open(save_path, "wb") as out_file:
            pkl.dump(self.features, out_file)

    def load_features(self, load_path="features.pkl"):
        with open(load_path,"rb") as in_file:
            self.features = pkl.load(in_file)

    def vectorize(self, text):
        vectorize_mtd = self.feature_method + "_vectorize"
        return getattr(self, vectorize_mtd)(text)
