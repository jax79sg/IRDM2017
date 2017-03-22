from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from HomeDepotCSVReader import HomeDepotReader
import FeatureEngineering
import numpy as np
from random import shuffle
import time

class Feature_Doc2Vec:

    def __trainModel(self, source_df, source_columnName, target_df, target_columnName):
        # Use PV-DM w/concatenation to preserve word ordering information, hierarchical sampling = 1 Reduce complexity from V sq
        model = Doc2Vec(dm=1, size=10, window=5, min_count=1, workers=4, alpha=0.025, dm_concat=1,
                        min_alpha=0.015, hs=1, negative=0)  # use fixed learning rate

        target_df['content'] = target_df['product_title'].map(str) + " " + \
                               target_df['product_description'].map(str) + " " + \
                               target_df['product_brand'].map(str)

        # print(target_df['content'].head(3))
        # Prepare the documents in gensim format
        docs = []
        for index, row in target_df.iterrows():
            # docs.append(TaggedDocument(FeatureEngineering.homedepotTokeniser(row.product_title), ['id_' + str(row.product_uid)]))
            docs.append(TaggedDocument(FeatureEngineering.homedepotTokeniser(row['content']),
                                       ['id_' + str(row.product_uid)]))

        target_df = target_df.drop('content', axis=1)
        # Build the vocab using gensim format
        model.build_vocab(docs)

        # Start training with random shuffle in every epoch
        EPOCH = 20
        for epoch in range(EPOCH):
            shuffle(docs)
            model.train(docs)

        # Save the model to disk
        model.save('models/doc2vec_trainedvocab.d2v')


    def getCosineSimilarity(self, source_df, source_columnName, target_df, target_columnName):
        start_time = time.time()
        try:
            # model = Doc2Vec.load('./models/doc2vec_' + target_columnName + '.d2v')
            model = Doc2Vec.load('models/doc2vec_trainedvocab.d2v')
        except FileNotFoundError:
            print("File not found. Do training. Takes 20min")
            self.__trainModel(source_df, source_columnName, target_df, target_columnName)

            model = Doc2Vec.load('models/doc2vec_trainedvocab.d2v')

        # print("[Feature_Doc2Vec] Loaded model: " + './doc2vec_' + target_columnName + '.d2v')
        print("Feature_Doc2Vec load model took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # tmp = source_df.search_term.map(lambda x: model.infer_vector(x,
        #                         alpha=0.025, min_alpha=0.025, steps=20))
        search_vectors = [np.array(model.infer_vector(FeatureEngineering.homedepotTokeniser(row), alpha=0.025, min_alpha=0.01, steps=20))
                          for _, row in source_df[source_columnName].iteritems()]
        print("Feature_Doc2Vec search_vectors took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        target_vectors = [np.array(model.infer_vector(
            FeatureEngineering.homedepotTokeniser(str(target_df[target_columnName].iloc[row].values[0])),
            alpha=0.025, min_alpha=0.01, steps=20))
                          for _, row in source_df.product_idx.iteritems()]
        print("Feature_Doc2Vec target_vectors took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # print("sourcedf: ", source_df[source_columnName])
        # print("targetdf: ", target_df[target_columnName])

        # for _, row in source_df.product_idx.iteritems():
        #     print("row: ", (target_df[target_columnName].iloc[row].values[0]))

        # print(cosine_similarity(search_vectors, target_vectors))

        result = []
        batch_size = 2000
        # Batch it
        for i in range(int(len(source_df)/batch_size)):
            inter_result = cosine_similarity(search_vectors[i * batch_size:(i * batch_size) + batch_size],
                                             target_vectors[i * batch_size:(i * batch_size) + batch_size])

            for i in range(inter_result.shape[0]):
                result.append(inter_result[i][i])

        # Compute the remaining
        start = int(len(source_df)/batch_size)*batch_size
        end = len(source_df)
        inter_result = cosine_similarity(search_vectors[start: end], target_vectors[start: end])

        # print("inter_result: \n", inter_result)
        # print("len of inter: ", len(inter_result))
        for j in range(inter_result.shape[0]):
            result.append(float(inter_result[j][j]))
        print("Feature_Doc2Vec Cosine Similarity took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # print("result: ", result)

        return result
        # s1 = ['duck', 'duck', 'duck']
        # v1 = model.infer_vector(s1, alpha=0.025, min_alpha=0.025, steps=20)
        # s2 = ['duck', 'duck', 'duck']
        # v2 = model.infer_vector(s2, alpha=0.025, min_alpha=0.025, steps=20)
        # # print(cosine_similarity(v1, search_vectors))



        # print([target_df[target_columnName].iloc[row] for _, row in source_df.product_idx.iteritems()])

        # print(model.most_similar('VersaBond'))

        # sentence = TaggedDocument(
        #     words=['Classic', 'Accessories', 'Tergggrazzo', 'Rectangular', 'Patio', 'Table', 'Cover'], tags=["SENT_0"])
        # sentence1 = TaggedDocument(
        #     words=['Classic', 'Accessories', 'Terrazzo', 'Rectangular', 'Patio', 'Table', 'Cover'], tags=["SENT_1"])

        # sentences = [sentence, sentence1]
        # print(model.similarity('Classic Accessories', 'Classic'))
        # print(model.wmdistance(sentence, sentence1))
        # print(model.score(sentence))
        # model.random.seed(0)



        # s1 = model.infer_vector(['Classic', 'Accessories', 'Terrazzo', 'Rectangular', 'Patio', 'Table', 'Cover'],
        #                         alpha=0.025, min_alpha=0.025, steps=20)
        # # s1 = model.infer_vector(['Rectangular', 'Table'],
        # #                         alpha=0.025, min_alpha=0.025, steps=5)
        # print(s1)
        # print("+++++++++++++++++++++++")
        # # model.random.seed(0)
        # s2 = model.infer_vector(['Classic', 'Accessories', 'Terrazzo', 'Rectangular', 'Patio', 'Table', 'Cover'],
        #                         alpha=0.035, min_alpha=0.025, steps=50)
        # # s1 = model.infer_vector(['Rectangular', 'Table'],
        # #                         alpha=0.025, min_alpha=0.025, steps=5)
        # print(s2)
        # print("+++++++++++++++++++++++")
        # # print(model.docvecs['id_113968'])
        # print("+++++++++++++++++++")
        # print(cosine_similarity(s1, model.docvecs['id_113968']))
        #
        # # print(cosine_similarity(s2, model.docvecs['id_113968']))

class LabeledLineSentence(object):
    def __init__(self, dataframe, columnName):
       self.dataframe = dataframe
       self.columnName = columnName

    def __iter__(self):
        for idx, row in self.dataframe.iterrows():
            yield TaggedDocument(FeatureEngineering.homedepotTokeniser(row.product_title), row.product_uid)


# #and change it=DocIt.DocIterator(data, docLabels) to
# it = LabeledLineSentence(data, docLabels)


if __name__ == "__main__":
    train_filename = '../../data/train.csv'
    test_filename = '../../data/test.csv'
    attribute_filename = '../../data/attributes.csv'
    description_filename = '../../data/product_descriptions.csv'

    reader = HomeDepotReader()

    train_query_df, product_df, attribute_df, test_query_df = reader.getQueryProductAttributeDataFrame(train_filename,
                                                                                                       test_filename,
                                                                                                       attribute_filename,
                                                                                                       description_filename)

    doc = Feature_Doc2Vec()
    doc.getCosineSimilarity(train_query_df, 'search_term', product_df, 'product_title')


    # documents = []
    # documents.append(TaggedDocument(['i', 'am', 'a', 'cat'], ['SENT_1']))
    # documents.append(TaggedDocument(['watching', 'a', 'movie'], ['SENT_2']))
    # documents.append(TaggedDocument(['doc2vec', 'rocks'], ['SENT_3']))
    #
    # model = Doc2Vec(size=10, window=8, min_count=0, workers=4)
    #
    # model.build_vocab(documents)
    # model.train(documents)
    #
    # search_phrase = ['i', 'am', 'a', 'cat']
    #
    # s1 = model.infer_vector(search_phrase, alpha=0.025, min_alpha=0.025, steps=20)
    #
    # print(cosine_similarity(s1, model.docvecs['SENT_1']))  # Print ~0.00795774
    #
    # s2 = model.infer_vector(['i', 'am', 'a', 'cat'], alpha=0.025, min_alpha=0.025, steps=20)
    #
    # print(cosine_similarity(s1, s2))  # Print ~0.9999882