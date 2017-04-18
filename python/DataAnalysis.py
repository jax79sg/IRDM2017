
from HomeDepotCSVReader import HomeDepotReader

train_filename = '../../data/train.csv'
test_filename = '../../data/test.csv'
attribute_filename = '../../data/attributes.csv'
description_filename = '../../data/product_descriptions.csv'

reader = HomeDepotReader()

train_query_df, product_df, attribute_df, test_query_df = reader.getQueryProductAttributeDataFrame(train_filename,
                                                                                                   test_filename,
                                                                                                   attribute_filename,
                                                                                                   description_filename)
print("train_query_df:" ,list(train_query_df))
print("product_df:", list(product_df))
print("attribute_df:", list(attribute_df))
print("test_query_df:", list(test_query_df))

groupBySearchTerm=train_query_df.groupby('search_term')
counter=0
oneOrMoreTermsInQuery=0
twoOrMoreTermsInQuery=0
minDocsReturn=9999999999
maxDocsReturn=0
for searchterm, eachGroup in groupBySearchTerm:
    eachSearchTerm=searchterm.split()
    if(len(eachSearchTerm)>1):
        twoOrMoreTermsInQuery = twoOrMoreTermsInQuery + 1
    elif(len(eachSearchTerm)==1):
        oneOrMoreTermsInQuery = oneOrMoreTermsInQuery + 1
    counter=counter+1
    noOfDocsForSearchTerm=eachGroup.shape[0]
    if(noOfDocsForSearchTerm>maxDocsReturn):
        maxDocsReturn=noOfDocsForSearchTerm
    if(noOfDocsForSearchTerm<minDocsReturn):
        minDocsReturn=noOfDocsForSearchTerm

print("oneOrMoreTermsInQuery:",oneOrMoreTermsInQuery)
print("twoOrMoreTermsInQuery:",twoOrMoreTermsInQuery)
print("minDocsReturn:",minDocsReturn)
print("maxDocsReturn:",maxDocsReturn)
print("No of queries:",counter)
print("No of products:", product_df.shape)

