#https://github.com/liheyuan/SimpleLMIR/blob/master/lmir.py converted to python 3
import re
from operator import itemgetter
import math
import urllib.parse

punc = ",./<>?;'\":\`~!@#$%^&*_-+=()"
table = str.maketrans(punc, " " * len(punc))


def analysis_text(text, stemming=False):
    # Lower and remove punc
    text = text.lower().translate(table)
    # Split by space
    return text.split()


def analysis_url(url):
    # Lower and remove /....
    url = urllib.parse.unquote(url)
    # Remove ( and )
    url = url.replace("(", "").replace(")", "")
    text = url.lower().translate(table)
    return text.split()


class Doc:
    def __init__(self, docid):
        self.docid = docid
        self.len = 0  # All word counts in doc
        self.words = {}  # Word and it's cnt

    def SetWords(self, words):
        for word in words:
            # Add tf of word
            self.words.setdefault(word, 0)
            self.words[word] += 1
        # Add doc len
        self.len = len(words)

    def GetDocTf(self, word, fuzzy=False):
        if not fuzzy:
            return self.words.get(word, 0)
        else:
            tf = 0
            for (key, cnt) in self.words.items():
                if word in key:
                    tf += cnt
            return tf

    def GetDocLen(self):
        return self.len

    def GetDocID(self):
        return self.docid


class Collection:
    def __init__(self):
        self.len = 0  # All word counts in collection
        self.words = {}

    def AddWords(self, words):
        for word in words:
            # Add tf of word
            self.words.setdefault(word, 0)
            self.words[word] += 1
        # Add doc len
        self.len += len(words)

    def GetCollTf(self, word, fuzzy=False):
        if not fuzzy:
            return self.words.get(word, 0)
        else:
            tf = 0
            for (key, cnt) in self.words.items():
                if word in key:
                    tf += cnt
            return tf

    def GetCollLen(self):
        return self.len


class LMIR:
    def __init__(self):
        self.coll = Collection()
        self.docs = []
        # Param lamada for LMIR.JM
        self.lmd_short = 0.65
        self.lmd_long = 0.25
        # Param u for DIR
        self.u = 2000

    def AddDocText(self, docid, text):
        # Analysis text words
        words = analysis_text(text)
        # Collection words
        self.coll.AddWords(words)
        # Doc words
        doc = Doc(docid)
        doc.SetWords(words)
        self.docs.append(doc)

    def AddDocUrl(self, docid, url):
        # Analysis url words
        words = analysis_url(url)
        # Collection words
        self.coll.AddWords(words)
        # Doc words
        doc = Doc(docid)
        doc.SetWords(words)
        self.docs.append(doc)

    # Rank doc in colls according to query, score by LMIR.JM
    def RankJM(self, query, fuzzy=False):
        # Analysis query words
        qws = analysis_text(query)
        # Choose lmd
        if len(qws) > 3:
            lmd = self.lmd_long
        else:
            lmd = self.lmd_short
        # Score each doc
        result = []
        for doc in self.docs:
            score = 1.0
            # score*=p(t|d)
            for qw in qws:
                ptd = float(doc.GetDocTf(qw, fuzzy)) / float(doc.GetDocLen())
                ptc = float(self.coll.GetCollTf(qw, fuzzy)) / float(self.coll.GetCollLen())
                pd = lmd * ptd + (1 - lmd) * ptc
                score *= pd
            # Add to result
            result.append((doc.GetDocID(), score))
        # Sort & return
        return sorted(result, key=itemgetter(1), reverse=True)

    # Rank doc in colls according to query, score by LMIR.DIR
    def RankDIR(self, query, fuzzy=False):
        # Analysis query words
        qws = analysis_text(query)
        # Score each doc
        result = []
        for doc in self.docs:
            score = 1.0
            # score*=p(t|d)
            for qw in qws:
                ptd_up = float(doc.GetDocTf(qw, fuzzy)) + float(self.u) * float(self.coll.GetCollTf(qw, fuzzy)) / float(
                    self.coll.GetCollLen())
                ptd_down = float(doc.GetDocLen()) + float(self.u)
                pd = float(ptd_up) / float(ptd_down)
                score *= pd
            # Add to result
            result.append((doc.GetDocID(), score))
        # Sort & return
        return sorted(result, key=itemgetter(1), reverse=True)

    # Rank doc in colls according to query, score by Kullback-Leibler Divergence(KLD)
    def RankKL(self, query, fuzzy=False):
        # Analysis query words
        qws = analysis_text(query)
        # Score each doc
        result = []
        for doc in self.docs:
            score = 0.0
            # score += -p(t|q)*log(P(t|d))
            for qw in qws:
                ptq = float(1) / float(len(qws))
                ptd = float(doc.GetDocTf(qw, fuzzy)) / float(doc.GetDocLen())
                if ptd == 0.0:
                    continue
                lptd = math.log(ptd, 2)
                score += -ptq * lptd
            # Add to result
            result.append((doc.GetDocID(), score))
        # Sort & return
        return sorted(result, key=itemgetter(1), reverse=True)

    def test(self):
        # Print collection
        print(self.coll.len)
        print(self.coll.words)

        # Print doc
        for doc in self.docs:
            print(doc.docid)
            print(doc.len)
            print(doc.words)



if __name__ == "__main__":
    # Test for text
    print("--------Test For Text--------")

    str1 = "I'm liheyuan from ict.\nWhat's your name?"
    str2 = "I'm coder4 from beijing.\nWhat's your name?"

    ir = LMIR()
    ir.AddDocText("doc1", str1)
    ir.AddDocText("doc2", str2)
    # ir.test()

    query = "coder4 name"
    print(ir.RankJM(query))
    print(ir.RankDIR(query))
    print(ir.RankKL(query))


    # Test for url
    print("--------Test For Url--------")

    url1 = "http://en.wikipedia.org/wiki/London_(disambiguation)"
    url2 = "http://en.wikipedia.org/wiki/Leviathan_(disambiguation)"
    url3 = "http://en.wikipedia.org/wiki/403(b)"

    ir2 = LMIR()
    ir2.AddDocUrl("url1", url1)
    ir2.AddDocUrl("url2", url2)
    ir2.AddDocUrl("url3", url3)
    # ir2.test()

    query = "403 wiki"
    print(ir2.RankJM(query, True) ) # Fuzzy
    print(ir2.RankDIR(query, True))
    print(ir2.RankKL(query, True))
