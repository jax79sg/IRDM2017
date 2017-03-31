import Feature_Word2Vec


class Word2VecQueryExpansion():
    """
    Changelog: 
    - 29/03 KS First committed                    
    Using Word Embeddings for Automatic Query Expansion
    General idea.
    For each query received, pass it through the Word2Vec model and pull out K
    vectors/words that's closest to it.
    Append these words to the original query and return as expanded query
    Note: Probably not suitable for classification based ranking models
    """
    def getExpandedQuery(self,querywords, maxNoOfAdditionalWords=1, minSimilarityLevel=0.7):
        """
        Changelog: 
        - 29/03 KS First committed                
        Return an expanded query based on similarity with pretrain word embeddings
        :param querywords: The words of original query e.g. "additional space"
        :param maxNoOfAdditionalWords: For each query word, how many words to expand.
        :param confidenceLevel: Min similarity level of expanded word 0 to 1.
        :return: 
        """
        expandedQuery=""
        w2v = Feature_Word2Vec.Feature_Word2Vec()
        for queryword in querywords.split( ):
            w2vSimilarwords=w2v.getSimilarWordVectors(queryword, maxNoOfAdditionalWords)
            # print("w2vSimilarwords:",w2vSimilarwords)
            expandedQuery = expandedQuery + " " + queryword
            for w2vSimilarword in w2vSimilarwords:
                # print("w2vSimilarword:",w2vSimilarword)
                # print("w2vSimilarword[0]:", w2vSimilarword[0])
                # print("w2vSimilarword[1]:", w2vSimilarword[1])

                word=w2vSimilarword[0]
                similarityscore = w2vSimilarword[1]
                if(similarityscore>=minSimilarityLevel):
                    expandedQuery=expandedQuery+" "+word

        return expandedQuery


class Doc2VecQueryExpansion():
    # TODO: To apply doc2vec.
    pass


if __name__ == "__main__":
    w2vExpand=Word2VecQueryExpansion()
    query="switch"
    print("Expanding query: ")
    print(w2vExpand.getExpandedQuery(query,maxNoOfAdditionalWords=2,minSimilarityLevel=0.65))