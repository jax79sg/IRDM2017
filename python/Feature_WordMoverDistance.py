import FeatureEngineering as fe
from HomeDepotCSVReader import HomeDepotReader
from gensim.models.keyedvectors import KeyedVectors

class Feature_WordMoverDistance():
    def __init__(self):
        # Load word2vec model using google pretrained vectors
        print("Loading Pre-trained vector")
        # Download link for GoogleNews-vectors-negative300.bin
        # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
        self.model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', binary=True)
        print("Loaded Pre-trained vector. Normalising now")
        # Normalize the embedding vectors
        self.model.init_sims(replace=True)

    def getDistance(self, source_df, source_columnName, target_df, target_columnName):
        print("Computing WordMoverDistance now")
        target_vectors = [self.model.wmdistance(
            fe.homedepotTokeniser(str(target_df[target_columnName].iloc[row.product_idx].values[0])),
            fe.homedepotTokeniser(str(row[source_columnName])))
                          for _, row in source_df.iterrows()]

        return target_vectors

if __name__ == "__main__":
    train_filename = '../../data/train_baby.csv'
    test_filename = '../../data/test_baby.csv'
    attribute_filename = '../../data/attributes.csv'
    description_filename = '../../data/product_descriptions.csv'

    reader = HomeDepotReader()

    train_query_df, product_df, attribute_df, test_query_df = reader.getQueryProductAttributeDataFrame(train_filename,
                                                  test_filename,
                                                  attribute_filename,
                                                  description_filename)


    wm = Feature_WordMoverDistance()

    train_query_df['wm_product_description'] = wm.getDistance(train_query_df, 'search_term', product_df, 'product_description')

    print(train_query_df['wm_product_description'].head(10))

