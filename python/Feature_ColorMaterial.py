import FeatureEngineering as fe
from HomeDepotCSVReader import HomeDepotReader
import pandas as pd
import numpy as np

# Color list adapted from
# https://kaggle2.blob.core.windows.net/forum-message-attachments/108037/3713/most_common_colors.txt
COLOR_LIST = (
"concrete",
"white",
"black",
"brown",
"gray",
"chrome",
"stainless steel",
"whites",
"red",
"browns",
"tans",
"bronze",
"silver",
"blacks",
"beige",
"stainless",
"blue",
"nickel",
"metallics",
"clear",
"grays",
"green",
"multi",
"beige",
"cream",
"tan",
"greens",
"yellow",
"wood",
"blues",
"reds",
"pinks",
"brushed nickel",
"orange",
"metallic",
"brass",
"yellows",
"golds",
"oil rubbed bronze",
"polished chrome",
"almond",
"multi-colored",
"dark brown wood",
"primed white",
"beige",
"bisque",
"biscuit",
"ivory",
"oranges",
"peaches",
"grey",
"unfinished wood",
"light brown wood",
"wood grain",
"silver metallic",
"copper",
"medium brown wood",
"soft white",
"gold",
"satin nickel",
"cherry",
"bright white",
"red",
"orange",
"teal",
"natural",
"oak",
"mahogany",
"aluminum",
"espresso",
"unfinished",
"purples",
"lavenders",
"brown",
"tan",
"steel",
"venetian bronze",
"slate",
"warm white",
"bone",
"pink",
"stainless look",
"reddish brown wood",
"solid colors",
"off-white",
"walnut",
"chocolate",
"light almond",
"vibrant brushed nickel",
"satin white",
"polished brass",
"linen",
"white primer",
"purple",
"charcoal",
"color",
"oil-rubbed bronze",
"melamine white",
"turquoises",
"aquas",
"blue",
"purple",
"primed",
"bisque",
"browns",
"tans",
"assorted colors",
"java",
"pewter",
"chestnut",
"yellow",
"gold",
"taupe",
"pacific white",
"cedar",
"monochromatic stainless steel",
"other",
"platinum",
"mocha",
"cream",
"sand",
"daylight",
"brushed stainless steel",
"powder-coat white",
)

class Feature_ColorMaterial():
    def __createColorMaterialColumn(self, s):
        c = []
        token = fe.homedepotTokeniser(s)

        for t in token:
            if t in COLOR_LIST:
                c.append(t)

        c = set(c)
        # print(str(c) + " ::::::::::::: \n" + str(s))
        return c


    def checkColorMaterialExists(self, train_query_df, product_df):
        product_df['content'] = product_df['product_title'].map(str) + " " + \
                                product_df['product_description'].map(str) + " " + \
                                product_df['attr_json'].map(str)

        # Create ColorMaterial Column
        print("Creating Color Column")
        product_df['product_color'] = product_df['content'].map(lambda x: self.__createColorMaterialColumn(x.lower()))
        print("Created Color Column")

        print(product_df.info())

        all = []
        for index, row in train_query_df.iterrows():
            token = fe.homedepotTokeniser(row['search_term'])
            product_color = product_df['product_color'].iloc[row.product_idx].values[0] #str(target_df[target_columnName].iloc[row].values[0])
            # print("product_color: ", product_color)
            col = []
            for t in token:
                if t in product_color:
                    col.append(t)

            # row['color'] = set(col)
            all.append(set(col))
            # print(str(row['search_term']) + " " + str(row['relevance']) +" $$$$$$$$$$$$$$$$$$$$$\n" + str(set(col)))
        product_df.pop('content')
        return all
        # train_query_df['search_term'].map(lambda x: self.__helperFunc(x.lower()))

    def __helperFunc(self, s):
        fe.homedepotTokeniser(s)


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

    col = Feature_ColorMaterial()
    train_query_df['color'] = col.checkColorMaterialExists(train_query_df, product_df)
    train_query_df['color_exist'] = train_query_df['color'].map(lambda x: 1 if len(x) > 0 else 0)
    train_query_df.color_exist = train_query_df.color_exist.astype(np.uint8)

    print(train_query_df.info())
    print(train_query_df.color.head(100))
    print(train_query_df.color_exist.head(100))

    train_query_df = pd.concat(
        [train_query_df, train_query_df.color.astype(str).str.strip('{}').str.get_dummies(', ').astype(np.uint8)], axis=1)

    updatedName = {}
    for i in list(train_query_df):
        if i[0] == "'":
            updatedName[i] = "color1hot_" + i.strip("''")

    train_query_df.rename(columns=updatedName, inplace=True)
    train_query_df.pop('set()')
    print(updatedName)

    # print(train_query_df["''"])

    print(train_query_df.info())

    print(train_query_df)
