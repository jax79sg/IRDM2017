# import enchant
from HomeDepotCSVReader import HomeDepotReader
from HomeDepotCSVWriter import HomeDepotCSVWriter
# import FeatureEngineering
import pandas as pd
import re
import csv
from collections import defaultdict
import numpy as np
import nltk
import string

def getSpellingCorrectionDict():
    spell_dict = eval(open('spelling_correction_dict.txt', 'r').read())
    return spell_dict




RE_D = re.compile('\d')
def hasDigits(string):
    return RE_D.search(string)

# tokeniser = re.compile("(?:[A-Za-z]{1,2}\.)+|[\w\']+|\?\!")
# def homedepotTokeniser(string):
#     return tokeniser.findall(string)

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

    vocab = defaultdict(int)
    for s in product_df.product_title:
        # no_punctuation = s.translate(None, string.punctuation)
        # token = nltk.word_tokenize(s)
        token = FeatureEngineering.homedepotTokeniser(s)
        # token = s.lower().split(' ')
        # print(s)
        for t in token:
            # print(t)
            vocab[str(t)] += 1

    for s in product_df.product_description:
        # token = s.lower().split(' ')
        # token = nltk.word_tokenize(s)
        token = FeatureEngineering.homedepotTokeniser(s)
        for t in token:
            vocab[str(t)] += 1

    brand_df = attribute_df[attribute_df.name == "MFG Brand Name"][['product_uid', 'value']]
    brand_df.rename(columns={'value': 'product_brand'}, inplace=True)
    product_df = pd.merge(product_df, brand_df, how='left', on='product_uid')

    for s in product_df.product_brand:
        # token = s.lower().split(' ')
        # token = nltk.word_tokenize(s)
        token = FeatureEngineering.homedepotTokeniser(str(s))
        for t in token:
            vocab[str(t)] += 1

    for s in attribute_df.value:
        # token = s.lower().split(' ')
        # token = nltk.word_tokenize(s)
        token = FeatureEngineering.homedepotTokeniser(str(s))
        for t in token:
            vocab[str(t)] += 1

    print(len(vocab))
    keys = list(vocab.keys())
    # print(keys[:100])

    df = pd.DataFrame(keys)

    # Dump home depot vocab to file
    HomeDepotCSVWriter().dumpCSV(df, "../../data/homedepot_vocab.txt", False)
    print("Dumped CSV")

    # Load the vocab from file
    d = enchant.DictWithPWL("en_US", '../../data/homedepot_vocab.txt')
    all_df = pd.concat((train_query_df, test_query_df), axis=0, ignore_index=True)
    n = 0
    dict = {}
    for s in all_df.search_term:
        token = s.split(" ")
        # print(str(n) + " " + str(d.check(token)))
        for t in token:
            if len(t)!= 0:
                if not hasDigits(t):
                    if d.check(t)==False:
                        if t not in dict:
                            suggest = d.suggest(t)
                            if len(suggest)>0:
                                dict[t] = suggest[0]
                                # print("'" + t + "' : '" + suggest[0] + "',")
                            else:
                                dict[t] = ''
                                # print("'" + t + "' : '',")

            n+=1

    print("Length all: ", n)
    print("Length of dict: ", len(dict))

    for k, v in dict.items():
        print("'" + k + "' : '" + v + "',")

    n = 0
    sortedkey = sorted(spell_dict)
    for k in sortedkey:
        if spell_dict[k].lower() != k.lower():
            print("'" + k.lower() + "' : '" + spell_dict[k].lower() + "',")
        else:
            n+=1

    for t in keys:
        if str(t).lower() in spell_dict:
            del spell_dict[str(t).lower()]
            n+=1

    print("Removed: ", n)






    d = enchant.Dict("en_US")
    # d = enchant.DictWithPWL("en_US", '../../data/homedepot_vocab.txt')
    # all_df = pd.concat((train_query_df, test_query_df), axis=0, ignore_index=True)
    n = 0
    count = 0
    dict = {}
    print("Starting spelling correction for product_description")
    for s in product_df.product_description:
        # if count == 10:
        #     break
        # count += 1
        token = homedepotTokeniser(s)
        # print(str(n) + " " + str(d.check(token)))
        for t in token:
            if len(t) != 0:
                if not hasDigits(t):
                    if d.check(t) == False:
                        if t not in dict:
                            suggest = d.suggest(t)
                            if len(suggest) > 0:
                                dict[t] = suggest[0]
                                # print("'" + t + "' : '" + suggest[0] + "',")
                            # else:
                            #     dict[t] = ''
                                # print("'" + t + "' : '',")

            n += 1

    print("Length all: ", n)
    print("Length of dict: ", len(dict))

    # for k, v in dict.items():
    #     print("'" + k + "' : '" + v + "',")

    n = 0
    f = open("product_description_spelling_correction.txt", "w")
    sortedkey = sorted(dict)
    for k in sortedkey:
        if dict[k].lower() != k.lower():
            print("'" + k.lower() + "' : '" + dict[k].lower() + "',")
            f.write(str("\"" + k.lower() + "\" : \"" + dict[k].lower() + "\",\n"))
        else:
            n += 1


    # f.write(str(spell_dict))
    f.close()
    # for t in keys:
    #     if str(t).lower() in spell_dict:
    #         del spell_dict[str(t).lower()]
    #         n += 1

    # for k in sortedkey:
    #     if str(t).lower() in spell_dict:
    #         del spell_dict[str(t).lower()]
    #         n += 1
    print("Removed: ", n)