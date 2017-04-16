import pandas as pd
# from nltk.stem.porter import *
# from nltk.stem.snowball import SnowballStemmer
import Stemmer
from Feature_TFIDF import Feature_TFIDF
from Feature_Doc2Vec import Feature_Doc2Vec
from Feature_BM25 import Feature_BM25
from Feature_ColorMaterial import Feature_ColorMaterial
from Feature_WordMoverDistance import Feature_WordMoverDistance
import time
from HomeDepotCSVReader import HomeDepotReader
from DataPreprocessing import DataPreprocessing
import numpy as np
import Feature_Spelling
import re
from nltk.corpus import stopwords
import nltk
from AutomaticQueryExpansion import Word2VecQueryExpansion
import Feature_Word2Vec
from Utilities import Utility

class HomeDepotFeature():
    def __init__(self):
        # self.stemmer = PorterStemmer()
        # self.stemmer = SnowballStemmer('english')
        self.stemmer = Stemmer.Stemmer('english')

    def getFeature(self, train_query_df, product_df, attribute_df, test_query_df,
                   features="brand,attribute,spelling,nonascii,stopwords,colorExist,color_onehot,brandExist,wmdistance,stemming,word2vec,Word2VecQueryExpansion,tfidf,tfidf_expandedquery,doc2vec,doc2vec_expandedquery,bm25,bm25expandedquery,doclength"):
        ## Please feel free to add feature into this method.
        ## For testing, you may want to comment out some feature generation to save time
        ## as some takes a long time to run.

        timetracker=Utility()
        if features.find("brand") != -1:
            # Create Brand Column
            product_df = self.__createBrandColumn(product_df, attribute_df)

        if features.find("attribute") != -1:
            # Create Attribute column as a JSON string
            # Column name is attr_json
            product_df = self.__createAttributeColumn(product_df, attribute_df)

        if features.find("spelling") != -1:
            # Perform spell correction on search_term
            print("Performing spell correction")
            spell_dict = Feature_Spelling.getSpellingCorrectionDict()
            # print(self.__spell_correction('lifeswivel', spell_dict))
            train_query_df['search_term'] = train_query_df['search_term'].map(
                lambda x: self.__spell_correction(x, spell_dict))
            product_df['product_description'] = product_df['product_description'].map(
                lambda x: self.__spell_correction(x, spell_dict))
            product_df['product_title'] = product_df['product_title'].map(
                lambda x: self.__spell_correction(x, spell_dict))
            product_df['attr_json'] = product_df['attr_json'].map(
                lambda x: self.__spell_correction(str(x), spell_dict))

        if features.find("nonascii") != -1:
            # Remove non-ascii characters
            print("Performing non-ascii removal")
            start_time = time.time()
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__nonascii_clean((x)))
            print("Non-ascii clean on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_title'] = product_df['product_title'].map(lambda x: self.__nonascii_clean(str(x)))
            print("Non-ascii clean on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # Run this to download the download the stopword list if you hit error
        # nltk.download()

        if features.find("stopwords") != -1:
            # Stopwords removal
            print("Performing stopwords removal")
            start_time = time.time()
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__stopword_removal((x)))
            print("stopwords removal on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_title'] = product_df['product_title'].map(lambda x: self.__stopword_removal(str(x)))
            print("stopwords removal on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_description'] = product_df['product_description'].map(lambda x: self.__stopword_removal(str(x)))
            print("stopwords removal on product_description took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['attr_json'] = product_df['attr_json'].map(lambda x: self.__stopword_removal(str(x)))
            print("stopwords removal on attr_jason took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find("colorExist") != -1:
            # Check if color in search_term exist in product_description column
            print("Performing color and material check")
            start_time = time.time()
            color = Feature_ColorMaterial()
            train_query_df['color'] = color.checkColorMaterialExists(train_query_df, product_df)
            train_query_df['color_exist'] = train_query_df['color'].map(lambda x: 1 if len(x)>0 else 0)
            # Save some memory. Change it to uint8
            train_query_df.color_exist = train_query_df.color_exist.astype(np.uint8)

            if features.find("color_onehot") != -1:
                train_query_df = self.__onehot_color(train_query_df)

            # Clean up unused column
            train_query_df.pop('color')
            print("Color and material check took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find("brandExist") != -1:
            # Check if brand in search term exist product_brand column
            print("Performing brand check")
            start_time = time.time()

            train_query_df['brand_exist'] = self.__brandExist(train_query_df, product_df)
            # train_query_df['brand_exist'] = train_query_df['search_term'].map(lambda x: 1 if len(x)>0 else 0)
            print("Brand check took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find('wmdistance') != -1:
            print("Performing Word Mover Distance")
            start_time = time.time()

            wm = Feature_WordMoverDistance()
            train_query_df['wm_product_description'] = wm.getDistance(train_query_df, 'search_term',
                                                                      product_df, 'product_description')
            print("WMDistance for product_description took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            train_query_df['wm_product_title'] = wm.getDistance(train_query_df, 'search_term',
                                                                      product_df, 'product_title')
            print("WMDistance for product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            train_query_df['wm_product_brand'] = wm.getDistance(train_query_df, 'search_term',
                                                                      product_df, 'product_brand')
            print("WMDistance for product_brand took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            train_query_df['wm_attr_json'] = wm.getDistance(train_query_df, 'search_term',
                                                                      product_df, 'attr_json')
            print("WMDistance for attr_json took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find("stemming") != -1:
            # # Stemming
            print("Performing Stemming")
            start_time = time.time()
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__stemming((x)))
            print("Stemming search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_title'] = product_df['product_title'].map(lambda x: self.__stemming(str(x)))
            print("Stemming product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_brand'] = product_df['product_brand'].map(lambda x: self.__stemming(str(x)))
            print("Stemming product_brand took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_description'] = product_df['product_description'].map(lambda x: self.__stemming(str(x)))
            print("Stemming product_description took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['attr_json'] = product_df['attr_json'].map(lambda x: self.__stemming(str(x)))
            print("Stemming attr_json took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find("word2vec") != -1:
            # Word2Vec
            print("===========Performing word2vec computation....this may take a while")
            timetracker.startTimeTrack()
            print("Merging product_title and description")
            print(list(product_df))
            product_df['content'] = product_df['product_title'].map(str) + " " + \
                                    product_df['product_description'].map(str) + " " + \
                                    product_df['product_brand'].map(str)
            timetracker.checkpointTimeTrack()
            print("Adding training query for that product id into the content")
            product_df = product_df.reset_index(drop=True)
            counter = 0
            for index, product in product_df.iterrows():
                # print("product:", product)
                productId = product['product_uid']
                # print("productId:",productId)
                df = train_query_df[train_query_df.product_uid == productId]
                # print("df:",df)
                searchterms = ""
                for index, row in df.iterrows():
                    searchterm = row['search_term']
                    searchterms = searchterms + " " + searchterm

                newString = product_df.iloc[counter]['content'] + " " + searchterms
                product_df.set_value(counter, 'content', newString)

                counter = counter + 1

            timetracker.checkpointTimeTrack()

            w2v = Feature_Word2Vec.Feature_Word2Vec()
            print("Convert DF into sentences for word2vec processing")
            sentences = w2v.convertDFIntoSentences(product_df, 'content')
            timetracker.checkpointTimeTrack()
            print("Training word2vec")
            w2v.trainModel(sentences)
            timetracker.checkpointTimeTrack()
            print("Validating...this should give some results like sofa")
            print(w2v.getVectorFromWord('stool'))
            print(w2v.getSimilarWordVectors('stool', 5))
            print("===========Completed word2vec computation")

        ##WARNING: This has to be before bm25expandedquery function call
        if features.find("Word2VecQueryExpansion") != -1:
            # Word2VecQueryExpansion
            print("===========Performing Word2VecQueryExpansion computation....this may take a super long time")
            timetracker.startTimeTrack()
            # print("Merging product_title and description")
            # print(list(product_df))
            # product_df['content']=product_df['product_title'].map(str) +" "+ \
            #                       product_df['product_description'].map(str) + " " + \
            #                       product_df['product_brand'].map(str)
            # product_df.head(1)
            print("Compute Word2VecQueryExpansion")
            w2cExpand = Word2VecQueryExpansion()
            timetracker.checkpointTimeTrack()
            # print("Remove merged column")
            # product_df=product_df.drop('content', axis=1)
            # For every training query-document pair, generate bm25
            print("Generate Word2VecQueryExpansion column")
            train_query_df = w2cExpand.computeExpandedQueryColumn(trainset=train_query_df,
                                                                  colName='Word2VecQueryExpansion')
            timetracker.checkpointTimeTrack()
            print("train_query_df:", list(train_query_df))
            print("train_query_df head:", train_query_df.head(1))
            print("Saving to csv")
            train_query_df.to_csv('../data.prune/train_query_with_Word2VecQueryExpansion.csv')
            timetracker.checkpointTimeTrack()
            print("===========Completed Word2VecQueryExpansion computation")

        if features.find("tfidf") != -1:
            # TF-IDF
            print("Performing TF-IDF")
            tfidf = Feature_TFIDF()
            train_query_df['tfidf_product_title'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_title')
            train_query_df['tfidf_product_brand'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_brand')
            train_query_df['tfidf_product_description'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_description')
            train_query_df['tfidf_attr_json'] = tfidf.getCosineSimilarity(train_query_df, 'search_term',
                                                                                    product_df,
                                                                                    'attr_json')
        if features.find("tfidf_expandedquery") != -1:
            # TF-IDF on expanded query
            print("Performing TF-IDF with expanded query")
            tfidf = Feature_TFIDF()
            train_query_df['tfidf_expanded_product_title'] = tfidf.getCosineSimilarity(train_query_df, 'Word2VecQueryExpansion', product_df,
                                                                              'product_title')
            train_query_df['tfidf_expanded_product_brand'] = tfidf.getCosineSimilarity(train_query_df, 'Word2VecQueryExpansion', product_df,
                                                                              'product_brand')
            train_query_df['tfidf_expanded_product_description'] = tfidf.getCosineSimilarity(train_query_df, 'Word2VecQueryExpansion', product_df,
                                                                              'product_description')
            train_query_df['tfidf_expanded_attr_json'] = tfidf.getCosineSimilarity(train_query_df, 'Word2VecQueryExpansion',
                                                                                    product_df,
                                                                                    'attr_json')

        if features.find("doc2vec") != -1:
            # Doc2Vec
            print("Performing Doc2Vec")
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_product_title'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_title')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_product_brand'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_brand')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_product_description'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_description')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_attr_json'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term',
                                                                                        product_df,
                                                                                        'attr_json')

        if features.find("doc2vec_expandedquery") != -1:
            # Doc2Vec
            print("Performing Doc2Vec with expanded query")
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_expanded_product_title'] = doc2vec.getCosineSimilarity(train_query_df,
                                                                                  'Word2VecQueryExpansion',
                                                                                  product_df,
                                                                                  'product_title')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_expanded_product_brand'] = doc2vec.getCosineSimilarity(train_query_df,
                                                                                  'Word2VecQueryExpansion',
                                                                                  product_df,
                                                                                  'product_brand')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_expanded_product_description'] = doc2vec.getCosineSimilarity(train_query_df,
                                                                                        'Word2VecQueryExpansion',
                                                                                        product_df,
                                                                                        'product_description')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_expanded_attr_json'] = doc2vec.getCosineSimilarity(train_query_df,
                                                                              'Word2VecQueryExpansion',
                                                                              product_df,
                                                                              'attr_json')

        if features.find("bm25") != -1:
            # BM25
            print("===========Performing BM25 computation....this may take a while")
            timetracker.startTimeTrack()
            print("Merging product_title and description")
            print(list(product_df))
            product_df['content']=product_df['product_title'].map(str) +" "+ \
                                  product_df['product_description'].map(str) + " " + \
                                  product_df['product_brand'].map(str)
            timetracker.checkpointTimeTrack()

            print("Adding training query for that product id into the content")
            product_df=product_df.reset_index(drop=True)
            counter=0
            for index,product in product_df.iterrows():
                # print("product:", product)
                productId=product['product_uid']
                # print("productId:",productId)
                df=train_query_df[train_query_df.product_uid==productId]
                # print("df:",df)
                searchterms=""
                for index,row in df.iterrows():
                    searchterm=row['search_term']
                    searchterms=searchterms+" "+searchterm

                newString=product_df.iloc[counter]['content']+" "+searchterms
                product_df.set_value(counter,'content',newString)

                counter=counter+1

            timetracker.checkpointTimeTrack()

            print("Compute BM25")
            bm25 = Feature_BM25(product_df)
            timetracker.checkpointTimeTrack()
            print("Remove merged column")
            product_df=product_df.drop('content', axis=1)
            #For every training query-document pair, generate bm25
            print("Generate bm25 column")
            train_query_df=bm25.computeBM25Column(trainset=train_query_df,destColName='bm25', searchTermColname='search_term')
            timetracker.checkpointTimeTrack()
            print("train_query_df:",list(train_query_df))
            print("train_query_df head:",train_query_df.head(1))
            print("Saving to csv")
            train_query_df.to_csv('../data.prune/train_query_with_bm25_search_term.csv')
            timetracker.checkpointTimeTrack()
            print("===========Completed BM25 computation")

        if features.find("bm25expandedquery") != -1:
            if features.find("Word2VecQueryExpansion") != -1:
                # bm25expandedquery
                print("===========Performing BM25expanded computation....this may take a while")
                timetracker.startTimeTrack()
                print("Merging product_title and description")
                print(list(product_df))
                product_df['content']=product_df['product_title'].map(str) +" "+ \
                                      product_df['product_description'].map(str) + " " + \
                                      product_df['product_brand'].map(str)
                product_df.head(1)
                timetracker.checkpointTimeTrack()

                print("Adding training query for that product id into the content")
                product_df = product_df.reset_index(drop=True)
                counter = 0
                for index, product in product_df.iterrows():
                    # print("product:", product)
                    productId = product['product_uid']
                    # print("productId:",productId)
                    df = train_query_df[train_query_df.product_uid == productId]
                    # print("df:",df)
                    searchterms = ""
                    for index, row in df.iterrows():
                        searchterm = row['search_term']
                        searchterms = searchterms + " " + searchterm

                    newString = product_df.iloc[counter]['content'] + " " + searchterms
                    product_df.set_value(counter, 'content', newString)

                    counter = counter + 1

                timetracker.checkpointTimeTrack()


                print("Compute BM25")
                bm25 = Feature_BM25(product_df)
                timetracker.checkpointTimeTrack()
                print("Remove merged column")
                product_df=product_df.drop('content', axis=1)
                #For every training query-document pair, generate bm25
                print("Generate bm25 column")
                train_query_df=bm25.computeBM25Column(trainset=train_query_df,destColName='bm25expandedquery', searchTermColname='Word2VecQueryExpansion')
                timetracker.checkpointTimeTrack()
                print("train_query_df:",list(train_query_df))
                print("train_query_df head:",train_query_df.head(1))
                print("Saving to csv")
                train_query_df.to_csv('../data.prune/train_query_with_bm25_Word2VecQueryExpansion.csv')
                timetracker.checkpointTimeTrack()
                print("===========Completed BM25expanded computation")
            else:
                print("ERROR: Cannot proceed with bm25expandedquery. Word2VecQueryExpansion is not enabled. It is a prerequisite of bm25expandedquery.")


        if features.find("bm25description") != -1:
            if features.find("Word2VecQueryExpansion") != -1:
                # bm25expandedquery
                print("===========Performing bm25description computation....this may take a while")
                timetracker.startTimeTrack()
                print(list(product_df))
                # product_df['content']=product_df['product_title'].map(str) +" "+ \
                #                       product_df['product_description'].map(str) + " " + \
                #                       product_df['product_brand'].map(str)
                product_df['content']=product_df['product_description'].map(str)

                product_df.head(1)
                timetracker.checkpointTimeTrack()

                print("Adding training query for that product id into the content")
                product_df = product_df.reset_index(drop=True)
                counter = 0
                for index, product in product_df.iterrows():
                    # print("product:", product)
                    productId = product['product_uid']
                    # print("productId:",productId)
                    df = train_query_df[train_query_df.product_uid == productId]
                    # print("df:",df)
                    searchterms = ""
                    for index, row in df.iterrows():
                        searchterm = row['search_term']
                        searchterms = searchterms + " " + searchterm

                    newString = product_df.iloc[counter]['content'] + " " + searchterms
                    product_df.set_value(counter, 'content', newString)

                    counter = counter + 1

                timetracker.checkpointTimeTrack()


                print("Compute BM25")
                bm25 = Feature_BM25(product_df)
                timetracker.checkpointTimeTrack()
                print("Remove merged column")
                product_df=product_df.drop('content', axis=1)
                #For every training query-document pair, generate bm25
                print("Generate bm25 column")
                train_query_df=bm25.computeBM25Column(trainset=train_query_df,destColName='bm25description', searchTermColname='Word2VecQueryExpansion')
                timetracker.checkpointTimeTrack()
                print("train_query_df:",list(train_query_df))
                print("train_query_df head:",train_query_df.head(1))
                print("Saving to csv")
                train_query_df.to_csv('../data.prune/train_query_with_bm25_Word2VecQueryExpansion.csv')
                timetracker.checkpointTimeTrack()
                print("===========Completed bm25description computation")
            else:
                print("ERROR: Cannot proceed with bm25description. Word2VecQueryExpansion is not enabled. It is a prerequisite of bm25expandedquery.")


        if features.find("bm25title") != -1:
            if features.find("Word2VecQueryExpansion") != -1:
                # bm25expandedquery
                print("===========Performing bm25title computation....this may take a while")
                timetracker.startTimeTrack()
                print(list(product_df))
                # product_df['content']=product_df['product_title'].map(str) +" "+ \
                #                       product_df['product_description'].map(str) + " " + \
                #                       product_df['product_brand'].map(str)
                product_df['content']=product_df['product_title'].map(str)

                product_df.head(1)
                timetracker.checkpointTimeTrack()

                print("Adding training query for that product id into the content")
                product_df = product_df.reset_index(drop=True)
                counter = 0
                for index, product in product_df.iterrows():
                    # print("product:", product)
                    productId = product['product_uid']
                    # print("productId:",productId)
                    df = train_query_df[train_query_df.product_uid == productId]
                    # print("df:",df)
                    searchterms = ""
                    for index, row in df.iterrows():
                        searchterm = row['search_term']
                        searchterms = searchterms + " " + searchterm

                    newString = product_df.iloc[counter]['content'] + " " + searchterms
                    product_df.set_value(counter, 'content', newString)

                    counter = counter + 1

                timetracker.checkpointTimeTrack()


                print("Compute BM25")
                bm25 = Feature_BM25(product_df)
                timetracker.checkpointTimeTrack()
                print("Remove merged column")
                product_df=product_df.drop('content', axis=1)
                #For every training query-document pair, generate bm25
                print("Generate bm25 column")
                train_query_df=bm25.computeBM25Column(trainset=train_query_df,destColName='bm25title', searchTermColname='Word2VecQueryExpansion')
                timetracker.checkpointTimeTrack()
                print("train_query_df:",list(train_query_df))
                print("train_query_df head:",train_query_df.head(1))
                print("Saving to csv")
                train_query_df.to_csv('../data.prune/train_query_with_bm25_Word2VecQueryExpansion.csv')
                timetracker.checkpointTimeTrack()
                print("===========Completed bm25title computation")
            else:
                print("ERROR: Cannot proceed with bm25title. Word2VecQueryExpansion is not enabled. It is a prerequisite of bm25expandedquery.")


        if features.find("bm25brand") != -1:
            if features.find("Word2VecQueryExpansion") != -1:
                # bm25expandedquery
                print("===========Performing bm25brand computation....this may take a while")
                timetracker.startTimeTrack()
                print(list(product_df))
                # product_df['content']=product_df['product_title'].map(str) +" "+ \
                #                       product_df['product_description'].map(str) + " " + \
                #                       product_df['product_brand'].map(str)
                product_df['content']=product_df['product_brand'].map(str)

                product_df.head(1)
                timetracker.checkpointTimeTrack()

                print("Adding training query for that product id into the content")
                product_df = product_df.reset_index(drop=True)
                counter = 0
                for index, product in product_df.iterrows():
                    # print("product:", product)
                    productId = product['product_uid']
                    # print("productId:",productId)
                    df = train_query_df[train_query_df.product_uid == productId]
                    # print("df:",df)
                    searchterms = ""
                    for index, row in df.iterrows():
                        searchterm = row['search_term']
                        searchterms = searchterms + " " + searchterm

                    newString = product_df.iloc[counter]['content'] + " " + searchterms
                    product_df.set_value(counter, 'content', newString)

                    counter = counter + 1

                timetracker.checkpointTimeTrack()


                print("Compute BM25")
                bm25 = Feature_BM25(product_df)
                timetracker.checkpointTimeTrack()
                print("Remove merged column")
                product_df=product_df.drop('content', axis=1)
                #For every training query-document pair, generate bm25
                print("Generate bm25 column")
                train_query_df=bm25.computeBM25Column(trainset=train_query_df,destColName='bm25brand', searchTermColname='Word2VecQueryExpansion')
                timetracker.checkpointTimeTrack()
                print("train_query_df:",list(train_query_df))
                print("train_query_df head:",train_query_df.head(1))
                print("Saving to csv")
                train_query_df.to_csv('../data.prune/train_query_with_bm25_Word2VecQueryExpansion.csv')
                timetracker.checkpointTimeTrack()
                print("===========Completed bm25brand computation")
            else:
                print("ERROR: Cannot proceed with bm25brand. Word2VecQueryExpansion is not enabled. It is a prerequisite of bm25expandedquery.")



        if features.find("doclength") != -1:
            # Document Length
            print("Performing Document Length")
            product_df['len_product_title'] = product_df['product_title'].map(lambda x: len(homedepotTokeniser(x)))
            train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_product_title']], how='left',
                                      on='product_uid')
            product_df['len_product_description'] = product_df['product_description'].map(lambda x: len(homedepotTokeniser(x)))
            train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_product_description']], how='left',
                                      on='product_uid')
            product_df['len_brand'] = product_df['product_brand'].map(lambda x: len(homedepotTokeniser(x)))
            train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_brand']], how='left',
                                      on='product_uid')
            train_query_df['len_search_term'] = train_query_df['search_term'].map(lambda x: len(homedepotTokeniser(x)))


        print("train_query_df final column:\n", train_query_df.info())

        return train_query_df,product_df,attribute_df

    def getFeature_old(self, df):
        ## Please feel free to add feature into this method.
        ## For testing, you may want to comment out some feature generation to save time
        ## as some takes a long time to run.

        # Create Brand Column
        # df = self.__createBrandColumn(df)

        #TODO: Chun Siong Working on Spell correction

        # Remove non-ascii characters
        print("Performing non-ascii removal")
        start_time = time.time()
        df['search_term'] = df['search_term'].map(lambda x: self.__nonascii_clean((x)))
        print("Non-ascii clean on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        df['product_title'] = df['product_title'].map(lambda x: self.__nonascii_clean(str(x)))
        print("Non-ascii clean on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))


        # # Stemming
        print("Performing Stemming")
        start_time = time.time()
        df['search_term'] = df['search_term'].map(lambda x: self.__stemming((x)))
        print("Stemming search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        df['product_title'] = df['product_title'].map(lambda x: self.__stemming(str(x)))
        print("Stemming product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        df['product_brand'] = df['product_brand'].map(lambda x: self.__stemming(str(x)))
        print("Stemming brand took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # TF-IDF
        print("Performing TF-IDF")
        tfidf = self.__create_TFIDF(df, "product_title")
        df['tfidf_product_title'] = tfidf
        tfidf = self.__create_TFIDF(df, "product_description")
        df['tfidf_product_description'] = tfidf
        tfidf = self.__create_TFIDF(df, "value")
        df['tfidf_attributes_value'] = tfidf
        tfidf = self.__create_TFIDF(df, "brand")
        df['tfidf_brand'] = tfidf

        # Document Length
        print("Performing Document Length")
        df['len_product_title'] = df['product_title'].map(lambda x: len(x.split()))
        df['len_product_description'] = df['product_description'].map(lambda x: len(x.split()))
        # df['len_brand'] = df['brand'].map(lambda x: len(str(x).split()))
        df['len_search_term'] = df['search_term'].map(lambda x: len(str(x).split()))

        print(df.info())

        return df

    def __createBrandColumn(self, product_df, attribute_df):
        brand_df = attribute_df[attribute_df.name == "MFG Brand Name"][['product_uid', 'value']]
        brand_df.rename(columns={'value': 'product_brand'}, inplace=True)
        product_df = pd.merge(product_df, brand_df, how='left', on='product_uid')
        # print("No. of product without brand: ", product_df.product_brand.isnull().sum())
        product_df.product_brand.replace(np.NaN, 'unknown_brand_value', inplace=True)
        return product_df

    def __createAttributeColumn(self, product_df, attribute_df):
        dp = DataPreprocessing()
        attribute_doc_df = dp.getAttributeDoc(attribute_df)
        # attribute_doc_df
        return product_df.join(attribute_doc_df.set_index('product_uid'), on='product_uid')

    def __spell_correction(self, s, spell_dict):
        #return " ".join([spell_dict[word.lower()] if word.lower() in spell_dict else word.lower()
        #                 for word in homedepotTokeniser(s)])

        return " ".join([spell_dict[word.lower()] if word.lower() in spell_dict else word
                         for word in homedepotTokeniser(s)])

    def __stemming(self, s):
        # return " ".join([self.stemmer.stemWord(word) for word in s.lower().split()])
        return " ".join([self.stemmer.stemWord(word.lower()) for word in homedepotTokeniser(s)])

    def __nonascii_clean(self,s):
        return "".join(letter for letter in s if ord(letter) < 128)

    def __stopword_removal(self, s):
        return " ".join([word for word in homedepotTokeniser(s) if not word in stopwords.words('english')])

    def __onehot_color(self, train_query_df):
        # Get color in one-hot fashion
        train_query_df = pd.concat(
            [train_query_df,
             train_query_df.color.astype(str).str.strip('{}').str.get_dummies(', ').astype(np.uint8)], axis=1)
        updatedName = {}
        for i in list(train_query_df):
            if i[0] == "'":
                updatedName[i] = "color1hot_" + i.strip("''")

        train_query_df.rename(columns=updatedName, inplace=True)
        train_query_df.pop('set()')

        return train_query_df

    def __brandExist(self, train_query_df, product_df):
        all = []
        for index, row in train_query_df.iterrows():
            product_brand = product_df['product_brand'].iloc[row.product_idx].values[0]

            if product_brand.lower() in row['search_term'].lower():
                all.append(1)
                # print(str(row['search_term']) + " ====== " + str(product_brand))
            else:
                all.append(0)

        return all

tokeniser = re.compile("(?:[A-Za-z]{1,2}\.)+|[\w\']+|\?\!")
def homedepotTokeniser(string):
    return tokeniser.findall(string)

if __name__ == "__main__":
    train_filename = '../../data/train.csv'
    test_filename = '../../data/test.csv'
    attribute_filename = '../../data/attributes.csv'
    description_filename = '../../data/product_descriptions.csv'

    reader = HomeDepotReader()
    train_df, test_df = reader.getMergedDataFrame(train_filename, test_filename, attribute_filename, description_filename)

    all_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)

    print("Starting Feature Engineering")
    feat = HomeDepotFeature()
    all_df = feat.getFeature(train_df,features="brand,spelling,nonascii,stopwords,stemming,tfidf,doc2vec,bm25,doclength")

    train_df = all_df.iloc[:train_df.shape[0]]
    test_df = all_df.iloc[train_df.shape[0]:]

