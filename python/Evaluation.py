
import numpy as np
import pandas as pd
import math


class NDCG_Eval():
    DCGmax=0
    RELEVANCE_SCORE_COL='relevance_int'
    DCGMAX_COL='dcg_max'
    NDCG_COL = 'ndcg_p'
    DCGP_COL = 'dcg_p'
    SEARCHTERM_COL='search_term'
    DOCID_COL='product_uid'
    def __init__(self):
        """
        Changelog:
        6/4 KS First commit
        Baseline idea is to be able to compare the ranking capability of each model using NDCG.
        Assumption: From gold dataset (Assuming that the dataset given has the correct relevance for query->docs. 
                    Our models will strictly return that same number of documents too.
        Steps:
        Pretrain
        1. Compute DCGmax
            -   For each query-> docS set in the gold dataset, 
                - Generate relevance score (Can reuse relevance_int for this)
                - Sort according to relevance score (Biggest to smallest)
                - Compute DCGp (Where p refers to number of docS in query)
                - Set DCGmax= DCGp
        
        Prediction step
        1. Compute NDCGp
            -   Use your model to make relevance predictions
            -   For each query-docS set
                - Sort according to your predicted relevance (Biggest to smallest)
                - Attach the relevance_score computed in PreTrain to each query-doc in the query-docs set
                    - E.g. In goldset. "hello" returns ("doc1,3"), ("doc2,1"), ("doc3,2")
                            Then your prediction for "hello" returns ("doc1,1"), ("doc2,1"), ("doc3,3")
                            After sorting "hello" returns ("doc3,3"), ("doc1,1"), ("doc2,1"), 
                            Now replace the relevance with the gold set relevance score "hello" returns ("doc3,2"), ("doc1,3"), ("doc2,1"),
                - Compute DCGp
                - Compute NDCG=DCGp/DCGmax
            -   Compute NDCGp = the mean of all NDCG across all queries.
            - The NDCGp can be compared against other Ranking models if they use the same dataset
            - Largest NDCPp represents better ranking
        """

    def computeAvgNDCG(self,goldDF,predictDF):
        """
        This will handle the NDCG computation based on the dataset format we have.
        :param goldDF: Required columns: search_term, product_uid, relevance_int
        :param predictDF: Required columns: search_term, product_uid, predicted_relevance
        :return: 
        """
        print("Filtering essential columns")
        goldDF=goldDF.filter(items=['relevance_int','search_term','product_uid'])
        predictDF = predictDF.filter(items=['relevance_int', 'search_term', 'product_uid'])
        print("goldDF columns:",list(goldDF))
        print("predictDF columns:", list(predictDF))
        print("Completed: Filtering essential columns")

        print("STARTING AVG_NDCG computation...this operation can take a while..")
        ##Pretrain
        ##1. Compute DCGmax

        #Sort by query small to big, relevance big to small
        print("Sorting by query small to big, relevance big to small")
        goldDF=goldDF.sort_values(['search_term',self.RELEVANCE_SCORE_COL,'product_uid'],ascending=[True,False,True])
        # print("goldDF sorted:\n",goldDF)
        goldDF[self.DCGMAX_COL]=-1
        groupByGoldDFSearchTerm=goldDF.groupby('search_term')
        for searchterm, eachGroup in groupByGoldDFSearchTerm:
            DCGmax=self._computeDCG(eachGroup[self.RELEVANCE_SCORE_COL].as_matrix())
            goldDF.ix[goldDF.search_term==searchterm,self.DCGMAX_COL]=DCGmax
        # goldDF.to_csv('../data/dcgMax.csv')
        # print("Creation of DCGmax:\n",goldDF)
        print("Completed: Sorting by query small to big, relevance big to small")

        ## Prediction step
        ## 1. Compute NDCGp

        ## Sort according to prediction relevance
        print("Sorting predictdf according to prediction relevance")
        predictDF = predictDF.sort_values(['search_term', self.RELEVANCE_SCORE_COL, 'product_uid'],ascending=[True, False, True])
        # print("predictDF sorted to predicted relevance:\n",predictDF)
        print("Completed: Sorting predictdf according to prediction relevance")

        ##- Attach the relevance_score computed in PreTrain to each query-doc in the query-docs set
        print("Attaching the relevance_score computed in PreTrain to each query-doc in the query-docs set")
        predictDF=predictDF.drop(self.RELEVANCE_SCORE_COL, axis=1)
        predictDF = pd.merge(predictDF,goldDF, on=[self.SEARCHTERM_COL, self.DOCID_COL])
        # print("predictDF after replacing with relevance scores\n:",predictDF)
        print("Completed: Attaching the relevance_score computed in PreTrain to each query-doc in the query-docs set")

        ##- Compute DCGp
        print("Computing DCGp of predicted sets")
        predictDF[self.DCGP_COL] = -1
        groupByPredictDFSearchTerm=predictDF.groupby('search_term')
        for searchterm, eachGroup in groupByPredictDFSearchTerm:
            DCGp=self._computeDCG(eachGroup[self.RELEVANCE_SCORE_COL].as_matrix())
            predictDF.ix[predictDF.search_term==searchterm,self.DCGP_COL]=DCGp
        # print("Creation of DCGp predict:\n",predictDF)
        print("Compeleted: Computing DCGp of predicted sets")


        ## Ensuring data correctness before final computation
        print("Ensuring data correctness before final computation")
        groupByPredictDFSearchTerm=predictDF.groupby('search_term')
        searchtermsDropped=[]
        for searchterm, eachGroup in groupByPredictDFSearchTerm:
            #Remove all DCGmax=0 entries, this happens if all document relevance for that query is zero.
            dcgMax=eachGroup[self.DCGMAX_COL].iloc[0]
            if (dcgMax==0):
                predictDF = predictDF[(predictDF.search_term != searchterm)]
                searchtermsDropped.append(searchterm)

        print("No of search terms removed from NDCG: ", len(searchtermsDropped),searchtermsDropped)
        print("Completed: Ensuring data correctness before final computation")


        ## - Compute
        ## NDCG = DCGp / DCGmax
        print("Computing NDCG = DCGp / DCGmax")
        def ndcg(columns):
            ndcg=columns[self.DCGP_COL]/columns[self.DCGMAX_COL]
            return ndcg

        # predictDF.to_csv('../data/zeroMax.csv')
        predictDF[self.NDCG_COL]=predictDF.apply(ndcg,axis=1)
        # print("Creation of NDCGp predict:\n", predictDF)
        print("Completed: Computing NDCG = DCGp / DCGmax")

        ## - Compute NDCGp = the mean of all NDCG across all queries.
        print("Computing avgNDCP = the mean of all NDCG across all queries.")
        groupByPredictDFSearchTerm = predictDF.groupby('search_term')
        totalNDCG=0
        count=0
        for searchterm, eachGroup in groupByPredictDFSearchTerm:
            NDCGp=eachGroup[self.NDCG_COL].iloc[0]
            totalNDCG=totalNDCG+NDCGp
            count=count+1

        print("totalNDCG:",totalNDCG)
        print("count:", count)
        avgNDCG=totalNDCG/count
        print("avgNDCG:",avgNDCG)
        return avgNDCG

    def _computeDCG(self, relevance_scores):
        """
        Changelog:
        6/4 KS First commit
        Compute the DCG based on 
        DCG= sum of (2^relevance_scores_i-1)/(log2(i+1), where i starts from 1 and stops at number of relevance scores.
        
        :param relevance_scores: An array of relevances_scores. Note: The order is important. 
                Make sure its ordered accordingly before using this function. See __init__ notes.
        :param isDCGmax Set DCGmax to the outcome of this computation                 
        :return: DCG: A postive float number (No bounds) .
        """

        n=relevance_scores.shape[0]

        i=1
        DCGp=0
        for relevance_score in relevance_scores:
            # print("relevance_score", relevance_score)
            gain=math.pow(2,relevance_score)-1
            # print("gain:",gain)
            discount=(math.log((i+1),2))
            discountedGain=gain/discount
            # print("discountedGain:",discountedGain)
            DCGp=DCGp+discountedGain
            i=i+1

        return DCGp

        
if __name__ == "__main__":

    relevance_scores=[2,1,0,0,2,1,0,1,0,0]
    np_relevance_scores=np.array(relevance_scores)
    ndcgEval=NDCG_Eval()
    # dcg=ndcgEval.computeDCG(np_relevance_scores)
    # print(dcg)

    goldList=[['two',2,10001],['two',1,10011],['one',2,10001],['one',1,10002],['one',0,10003],['one',0,10004],['one',2,10005],['one',1,10006],['one',0,10007],['one',1,10008],['one',0,10009],['one',0,10010]]
    golddf=pd.DataFrame(goldList,columns=['search_term','relevance_int','product_uid'])

    predictList=[['two',3,10001],['two',3,10011],['one',1,10001],['one',2,10002],['one',1,10003],['one',0,10004],['one',0,10005],['one',2,10006],['one',1,10007],['one',2,10008],['one',2,10009],['one',1,10010]]
    predictdf = pd.DataFrame(predictList, columns=['search_term', 'relevance_int', 'product_uid'])
    # print(predictdf)
    avgNDCG=ndcgEval.computeAvgNDCG(golddf,predictdf)
    # print("avgNDCG for Ranker:",avgNDCG)
