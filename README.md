# IRDM2017
Information Retreival and Data Mining


**TODO for Data Processing:**
+ Stemming + lower case (Chun Siong: Done)
+ Spelling correction (Chun Siong: Done)
+ Remove punctuation (Chun Siong: Done. There is a tokeniser based on RE under FeatureEngineering.py)
+ Remove Non-ASCII (Kah Siong: Done)
+ Stopword removal (Min: Done)
+ Merge all attributes key value pair into a single text field (Min: Done)

**TODO for Feature Engineering:**
More ideas of features can be found here https://www.microsoft.com/en-us/research/project/mslr/
+ Query-independant (Document only)
    + Document Length (Chun Siong: Done)
    + Brand Column (Chun Siong: Done)
+ Query-dependant (Document and query)
    + TF-IDF (Chun Siong: Done)
    + BM25 on product title and description combined (Kah Siong: Done)
    + LMIR.ABS (Min: WIP, Feature_LMIR.py implemented, but not incorporated into FeatureEngineering.py)
    + LMIR.DIR
    + LMIR.JM
    + LDA (Chun Siong: Reading)
+ Query-only
    + Query expansion (Kah Siong: Done with Word2Vec Query Expansion)

+ Similarity measure
    + Cosine Similarity (Chun Siong: Done)
+ Doc2Vec (Chun Siong: Done)
+ Word2Vec (Kah Siong: Done)
+ KL ? (What's this?) Kulback lieber, i've seen it mentioned in comparisons which include BM25, LMIR, KL https://www.microsoft.com/en-us/research/publication/relevance-ranking-using-kernels/
+ Output all computed feature to Ranklib format


**TODO for Model Selection:**
+ Pointwise
    + Ordinal Regression (!!! Emine emailed Linear Regression cannot use libraries....Kah Siong: Training-Done, Prediction-Done, Evaluation-WIP) (MORD, and if MORD doesn't work then https://gist.github.com/agramfort/2071994)
    + Support Vector Machine 
    + Boosted Regression
    + Perception <- perceptron? 

+ Pairwise
    + RankNet (RankLib)
    + RankBoost (RankLib)
    + Coordinate Ascent (RankLib)
    + LambdaMart (RankLib)
    + MART (RankLib)
    + Random Forests (RankLib)

+ Listwise
    + ListNet (RankLib)
    + AdaRank (RankLib)

+ Neural network
    + Match tensor - (Min: Reading on this https://arxiv.org/pdf/1701.07795.pdf)

**TODO for Evaluation:**
+ NDCG (Kah Siong: Python Done) Note: libRank has in-built implementation for java
+ RMSE

**Goal**

Predict a relevance score for the provided combinations of search terms and products.

id, relevance

where id is id of test sample, relevance (score) is a value between 1 and 3..

**Important note to raters.**

The relevance score was made by the following considerations

- Examine the _Product Description_
- Do not only use the _product title_ to determine relevancy
- Focus on the following _attributes _when comparing the product to the query (E.g. Brand, Material, and Functionality)
- Brand is as important as the Functionality!

**Glossary**

| Items | Explanation |
| --- | --- |
| Relevance | Real number between 1 (not relevant) to 3 (relevant). (E.g. Can be 1.33) |
| Search/Product pairs | Search-Product pairsOne or more search terms that produced a product. |
| Evaluations | Search-Product-RelevanceOne or more search terms that produced a product. And this match was evaluated at a relevance defined above. |
| Id | Unique id to identify rows on the train or test sets. |
| Product\_uid | Id to identify the product. Product may appear more than once on the records |
| product\_title | Short title of the product. |
| search\_term | One or more search words separated by a space. |
| product\_description | A long description of the product (Think in terms of commercial) |
| Name | Name of an attribute (E.g. Material). This should be used with &#39;value&#39;. |
| Value | Value of an attribute (E.g. A value of a &#39;Material&#39; attribute is &#39;Steel&#39; |



**Datasets**

| File | Description | Sample |
| --- | --- | --- |
| Train.csv | Our training set, contains products, searches, and relevance scores. | Id, product\_uid, product\_title, search\_term, relevance1,100001,Husky 18 in. Total Tech Bag , husky tool bag, 3 2,100002, Vigoro 60 ft. No-Dig Edging, landscape edging, 2.67 |
| Test.csv | Same as training set, except no relevance. | Id, product\_uid, product\_title, search\_term1,100001, &quot;Husky 18 in. Total Tech Bag , husky tool bag&quot; 2,100002, &quot;Vigoro 60 ft. No-Dig Edging&quot;, &quot;landscape edging&quot; |
| Product\_descriptions.csv | Description of each product. | &quot;product\_uid&quot;,&quot;product\_description&quot;100001,&quot;Not only do angles make joints stronger, they also provide more consistent, straight corners. …. SD screws&quot; |
| Attributes.csv | provides extended information about a subset of the products (typically representing detailed technical specifications). Not every product will have attributes.  | &quot;product\_uid&quot;,&quot;name&quot;,&quot;value&quot;100001,&quot;Bullet01&quot;,&quot;Versatile connector for various 90° connections and home repair projects&quot; 100001,&quot;Material&quot;,&quot;Galvanized Steel&quot; |
