import pandas as pd


class HomeDepotCSVWriter:
    def write2RankLibCSV(self, df, filename):
        # TODO: Dump the training feature according to the formate in the URL
        # https://sourceforge.net/p/lemur/wiki/RankLib%20File%20Format/
        # http://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html
        raise NotImplemented

    def dumpCSV(self, df, filename, header=True, encoding=''):
        pd.DataFrame(df).to_csv(filename, index=False, header=header,encoding=encoding)
