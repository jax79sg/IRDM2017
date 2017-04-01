from gensim.models import Word2Vec
import Utilities
from pathlib import Path
from HomeDepotCSVReader import HomeDepotReader

class Feature_Word2Vec():
    modelFilename=None
    utility=None
    model=None

    def __init__(self, modelFilename='model/word2vec.model'):
        """
        Changelog: 
        - 29/03 KS First committed
        Initialise the class        
        :param modelFilename: 
        """
        self.utility=Utilities.Utility()
        self.modelFilename=modelFilename
        self._loadModel(self.modelFilename)


    def _loadModel(self,modelFilename):
        """
        Changelog: 
        - 29/03 KS First committed      
          Hidden method for init only. Loading of existing word2vec file
        :param modelFilename: 
        :return: 
        """
        my_file = Path(modelFilename)
        if my_file.is_file():
            print("Loading Word2Vec model")
            self.model = Word2Vec.load(self.modelFilename)
        else:
            print("Word2Vec model file not found, please run trainModel method")


    def trainModel(self, sentences):
        """
        Changelog: 
        - 29/03 KS First committed
        Train the sentences into vectors using word2Vec algorithm using gensim libraries.
        https://radimrehurek.com/gensim/models/word2vec.html
        Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013
        Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.        
        :param sentences: Format must be like this: [['This','is','one','sentence'],['This','is','another','sentence']]
        :return: 
        """
        self.utility.startTimeTrack()
        print("Initialise/Train Word2Vec model")
        self.model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, iter=5)
        self.utility.checkpointTimeTrack()

        print("Saving vectors to disk")
        self.model.save(self.modelFilename)
        self.utility.checkpointTimeTrack()

    def getVectorFromWord(self, word):
        """
        Changelog: 
        - 29/03 KS First committed        
        Return the vector for the given word
        :param word: 
        :return: 
        """
        return self.model.wv[word]

    def trainMoreWords(self,sentences):
        """
        Changelog: 
        - 29/03 KS First committed        
        If model has been previously trained, can add more words into training
        :param sentences:Format must be like this: [['This','is','one','sentence'],['This','is','another','sentence']]
        :return: 
        """
        print("Loading Word2Vec model")
        self.model = Word2Vec.load(self.modelFilename)
        print("Train Word2Vec model")
        self.model.build_vocab(sentences=sentences)
        self.model.train(sentences=sentences)
        print("Saving vectors to disk")
        self.model.save(self.modelFilename)
        self.utility.checkpointTimeTrack()

    def getSimilarWordVectors(self,word,noOfWordToReturn=5, retrain=False):
        """
        Changelog: 
        - 29/03 KS First committed        
        Returns words and their vectors thats closest to the given word.
        :param words: A single word
        :param noOfWordToReturn: Number of closest words to return. First one is closest.
        :param retrain: If encounter OOV, retrain the word2vec
        :return: 
        """
        result=[('',0)]
        try:
            result=self.model.most_similar(word, [], noOfWordToReturn)
        except KeyError:
            if(retrain):
                print('Word not found, retraining...')
                self.model.build_vocab([word])
                self.model.train([word])
                result = self.model.most_similar(word, [], noOfWordToReturn)
                print(result)
            else:
                print(word," is not in word2vec vocab. Returning empty")
        return result

    def convertDFIntoSentences(self,dataframe,columnName):
        """
        Changelog: 
        - 29/03 KS First committed                
        Take in a dataframe and converting into sentence for training.
        :param dataframe: 
        :param columnName: 
        :return: 
        """
        sentences=[]
        for row in dataframe[columnName]:
            #Removing commas - To remove situations of 'door' and 'door,' being similar

            rowSentences=row.split('.')
            for rowSentence in rowSentences:
                rowWords=rowSentence.split( )
                # print("rowWords---:",rowWords)
                strippedRowWords=[]
                for rowWord in rowWords:
                    strippedRowWords.append(rowWord.strip(','))

                sentences.append(strippedRowWords)
        print(sentences)
        return sentences

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
    print("train_query_df:",list(train_query_df))
    print("product_df:", list(product_df))
    print("attribute_df:", list(attribute_df))
    print("test_query_df:", list(test_query_df))

    product_df['content'] = product_df['product_title'].map(str) + " " + \
                            product_df['product_description'].map(str)

    w2v=Feature_Word2Vec()
    sentences=w2v.convertDFIntoSentences(product_df,'content')
    w2v.trainModel(sentences)
    print(w2v.getVectorFromWord('stool'))
    print(w2v.getSimilarWordVectors('stool',5))



