# UCL IRDM2017 Group 5
Information Retreival and Data Mining

## Manuals
** Generate features:**
1. Warning: This could take half a day to complete.
2. Ensure datasets data.original/attributes.csv, data.original/product_descriptions.csv, data.original/test.csv and data.original/train.csv are available.
3. Set the desired features to generate in desiredFeatures variable of RunMe.py
4. Run RunMe.py
5. Generated csv will be located in data/features_full.csv

** Train and generate results for Ordinal Regression:**
1. Ensure featureset data/features_doc2vec_sense2vec_pmi_20170418.csv is available.
2. If using a different featureset, please change file reference in OrdinalRegressionRanker.py (myFeatureSetFileReference variable)
3. Run OrdinalRegressionRanker.py
4. Results will be generated on screen or ordinal_private.RMSE_NDCG_.csv and ordinal_public.RMSE_NDCG_.csv

** Train and generate results for DNN:**
1. Ensure featureset data/features_doc2vec_sense2vec_20170416.csv is available.
2. If using a different featureset, please change file reference in DNN.ipynb (full_features_filename variable)
3. Run DNN.ipynb 
4. RMSE results will be generated on screen and results can be output to file specified.

** Train and generate results for XGBoost:**
1. Ensure featureset data/features_final_20170419.csv is available.
2. If using a different featureset, please change file reference in XGBoostRanker.py line 263
3. Run XGBoostRanker.py
4. RMSE results will be generated on screen and results.csv are output to file specified.

** Generate ensemble results:**
1. Ensure public and private test sets predictions are available in the format 'id','pred_revelance' columns.
2. Run ensemble.ipynb to generate ensemble prediction csv and RMSE, NDCG scores on screen.

** Generate features – for Random Forest & Bagging Algorithms:**
1. Ensure datasets attributes.csv, product_descriptions.csvtest.csv and train.csv are available.
2. Run RandomForestRanker.ipynb
3. The Generated csv will be located in data/features_rf_bag_lg.csv

** Train and generate results for Random Forest & Bagging Algorithms:**
1. Ensure featureset data/ features_rf_bag_lg.csv is available. (optional) .If using a different featureset, uncomment the appropriate line or if constructed from scratch please change file reference in RandomForestRanker.ipynb (df_full_clean.to_csv('features_rf_bag_lg.csv',index=False))
2. Run RandomForestRanker.ipynb
3. RMSE results for the selected algorithms (either Random Forest, Bagging or Logistic Regression) will be generated on screen and results can be output to file path specified.


## Ideas list
**Ideas for Data Processing:**
+ Stemming + lower case (Chun Siong: Done)
+ Spelling correction (Chun Siong: Done)
+ Remove punctuation (Chun Siong: Done. There is a tokeniser based on RE under FeatureEngineering.py)
+ Remove Non-ASCII (Kah Siong: Done)
+ Stopword removal (Min: Done)
+ Merge all attributes key value pair into a single text field (Min: Done)
+ Brand Column (Chun Siong: Done)
+ Color and material Column (Chun Siong: Done)

**Ideas for Feature Engineering:**
More ideas of features can be found here https://www.microsoft.com/en-us/research/project/mslr/
+ Query-independant (Document only)
    + Document Length (Chun Siong: Done)
    + Brand Column (Chun Siong: Done)
    + Document, search term, title length (Chun Siong: Done)
+ Query-dependant (Document and query)
    + TF-IDF (Chun Siong: Done)
    + Binary indicator if color/material in search term is also in product (Chun Siong: Done)
    + Binary indicator if brand in search term is also product brand (Chun Siong: Done)
    + BM25 on product title and description combined (Kah Siong: Done)
    + LMIR.ABS (Min: WIP, Feature_LMIR.py implemented, but not incorporated into FeatureEngineering.py)
    + LMIR.DIR
    + LMIR.JM
    + LDA
    + PMI (Kah Siong: Done)
    + Sense2vec (Min: Done)
    + Productuid (Min: Done)
    + spaCy noun chunks (Min: Done)


+ Cosine Similarity (Chun Siong: Done)
+ Doc2Vec (Chun Siong: Done)
+ Word2Vec (Kah Siong: Done)
+ Query expansion (Kah Siong: Done with Word2Vec Query Expansion)
+ KL ? (What's this?) Kulback lieber, i've seen it mentioned in comparisons which include BM25, LMIR, KL https://www.microsoft.com/en-us/research/publication/relevance-ranking-using-kernels/
+ Output all computed feature to Ranklib format

**Ideas for Model Selection:**

+ Pointwise
    + Logistic Regression (Kah Siong: Done)
    + Ordinal Regression and variants (LAD, LOGIT, LOGAT) (Kah Siong: Done) (Stick with Ridge variant best performer) (MORD, and if MORD doesn't work then https://gist.github.com/agramfort/2071994)
    + Factorisation Machine multiclass classifier (Kah Siong: Done) (NOT GOOD..Its running now but it doesn't seem to predict properly for one vs all multiclass... above 1 RMSE)
    + Support Vector Machine 
    + Boosted Regression
    + Perception <- perceptron? 
    + Gradient Boosted Regression Trees (Chun Siong: Done)
    + Deep learning methods
        + RNN - Match tensor https://arxiv.org/pdf/1701.07795.pdf - (Min: Done)
        + CNN -  (Min: Done)
        + DNN -  (Min: Done)

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

+ Ensemble 
    + weighted ensemble (Min: Done)
    
**Ideas for Evaluation:**
+ NDCG (Kah Siong: Done to accommodate our datasets)
+ RMSE

## Goal

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
