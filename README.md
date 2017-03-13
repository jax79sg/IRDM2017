# IRDM2017
Information Retreival and Data Mining


**TODO for Feature Engineering:**
+ Stemming (Chun Siong: Done)
+ TF-IDF (Chun Siong: WIP to improve performance)
+ Spelling correction (Chun Siong: Work in progress)
+ Document Length (Chun Siong: Done)
+ BM25 
+ Cosine Similarity (Chun Siong: Done)
+ Other similarity measure to compare TF-IDF/BM25
+ Remove punctuation
+ Remove Non-ASCII (Kah Siong: WIP)
+ Brand Column (Chun Siong: Done)
+ Merge all attributes key value pair into a single text field
+ LMIR.ABS
+ LMIR.DIR
+ LMIR.JM

**TODO for Model Selection:**
+ Pointwise
+ Pairwise


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
| Train.csv | Our training set, contains products, searches, and relevance scores . | Id, product\_uid, product\_title, search\_term, relevance1,100001,Husky 18 in. Total Tech Bag , husky tool bag, 3 2,100002, Vigoro 60 ft. No-Dig Edging, landscape edging, 2.67 |
| Test.csv | Same as training set, except no relevance. | Id, product\_uid, product\_title, search\_term1,100001, &quot;Husky 18 in. Total Tech Bag , husky tool bag&quot; 2,100002, &quot;Vigoro 60 ft. No-Dig Edging&quot;, &quot;landscape edging&quot; |
| Product\_descriptions.csv | Description of each product. | &quot;product\_uid&quot;,&quot;product\_description&quot;100001,&quot;Not only do angles make joints stronger, they also provide more consistent, straight corners. …. SD screws&quot; |
| Attributes.csv | provides extended information about a subset of the products (typically representing detailed technical specifications). Not every product will have attributes.  | &quot;product\_uid&quot;,&quot;name&quot;,&quot;value&quot;100001,&quot;Bullet01&quot;,&quot;Versatile connector for various 90° connections and home repair projects&quot; 100001,&quot;Material&quot;,&quot;Galvanized Steel&quot; |
