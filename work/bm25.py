import math
import jieba
text = '''
自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
所以它与语言学的研究有着密切的联系，但又有重要的区别。
自然语言处理并不是一般地研究自然语言，
而在于研制能有效地实现自然语言通信的计算机系统，
特别是其中的软件系统。因而它是计算机科学的一部分。
'''


def BM25(object):
     def __ini__(self, docs):
         self.D = len(docs)
         self.avgdl = sum(len(doc)+0.0 for doc in docs) / self.D
         self.docs = docs
         self.f = []
         self.df = {}
         self.idf = {}
         self.k1 = 1.5
         self.k2 = 0.75
         self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for w in doc:
                tmp[w] = tmp.get(w,0)+1
            # term frequency
            self.f.append(tmp)
            # doc frequency
            for k in tmp.keys():
                self.df[k] = self.df.get(k,0)+1
        for k,v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)- math.log(v+0.5)

    def sim(self, doc, index):
        score  = 0
        for w in doc:
            if w not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[w]*self.f[index][w]*(self.k1+1)
                      / (self.f[index][w]+self.k1*(1-self.b+self.b*d/self.avgdl)))
        return score

    def simall(self,doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


