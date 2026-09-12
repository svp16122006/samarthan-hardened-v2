"""Dependency-light TF-IDF + multinomial logistic regression for the Samarthan POC.
Designed to run on Python 3.14 without NumPy/SciPy/scikit-learn native extensions.
Engineering/demo model only; not clinical validation.
"""
from __future__ import annotations
import math, re, pickle
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"(?u)\b\w+\b")

class TfidfVectorizerLite:
    def __init__(self, ngram_range=(1,2), min_df=1):
        self.ngram_range=ngram_range; self.min_df=min_df
        self.vocabulary_={}; self.idf_=[]
    def _tokens(self, text):
        toks=TOKEN_RE.findall(text.lower())
        out=[]
        for n in range(self.ngram_range[0], self.ngram_range[1]+1):
            out += [' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)]
        return out
    def fit(self, texts):
        df=Counter(); total=len(texts)
        for t in texts:
            df.update(set(self._tokens(t)))
        terms=sorted(k for k,v in df.items() if v>=self.min_df)
        self.vocabulary_={t:i for i,t in enumerate(terms)}
        self.idf_=[math.log((1+total)/(1+df[t]))+1 for t in terms]
        return self
    def transform(self,texts):
        rows=[]
        for text in texts:
            counts=Counter(self._tokens(text)); row={}
            for term,c in counts.items():
                j=self.vocabulary_.get(term)
                if j is not None: row[j]=(1+math.log(c))*self.idf_[j]
            norm=math.sqrt(sum(v*v for v in row.values())) or 1.0
            rows.append({j:v/norm for j,v in row.items()})
        return rows
    def fit_transform(self,texts): return self.fit(texts).transform(texts)
    def get_feature_names_out(self):
        out=['']*len(self.vocabulary_)
        for t,i in self.vocabulary_.items(): out[i]=t
        return out

class LogisticRegressionLite:
    def __init__(self, classes, max_iter=800, lr=0.35, l2=0.002, seed=42):
        self.classes_=list(classes); self.max_iter=max_iter; self.lr=lr; self.l2=l2; self.seed=seed
        self.coef_=[]; self.intercept_=[]
    def _softmax(self, z):
        m=max(z); ex=[math.exp(min(50,max(-50,v-m))) for v in z]; s=sum(ex); return [v/s for v in ex]
    def fit(self, X, y):
        k=len(self.classes_); d=max((max(r.keys()) if r else -1 for r in X), default=-1)+1
        self.coef_=[[0.0]*d for _ in range(k)]; self.intercept_=[0.0]*k
        idx={c:i for i,c in enumerate(self.classes_)}
        n=len(X)
        for epoch in range(self.max_iter):
            gW=[[0.0]*d for _ in range(k)]; gb=[0.0]*k
            for row,label in zip(X,y):
                logits=[]
                for c in range(k): logits.append(self.intercept_[c]+sum(self.coef_[c][j]*v for j,v in row.items()))
                p=self._softmax(logits); yi=idx[label]
                for c in range(k):
                    e=p[c]-(1.0 if c==yi else 0.0); gb[c]+=e
                    for j,v in row.items(): gW[c][j]+=e*v
            # decaying step gives stable convergence on small synthetic set
            step=self.lr/(1.0+epoch*0.012)
            for c in range(k):
                self.intercept_[c]-=step*gb[c]/n
                for j in range(d):
                    self.coef_[c][j]-=step*(gW[c][j]/n+self.l2*self.coef_[c][j])
        return self
    def predict_proba(self,X):
        out=[]
        for row in X:
            z=[self.intercept_[c]+sum(self.coef_[c][j]*v for j,v in row.items()) for c in range(len(self.classes_))]
            out.append(self._softmax(z))
        return out
    def predict(self,X):
        probs=self.predict_proba(X); return [self.classes_[max(range(len(self.classes_)), key=lambda i:p[i])] for p in probs]

class TextRiskModel:
    def __init__(self, vectorizer, classifier): self.tfidf=vectorizer; self.clf=classifier
    def fit(self,texts,labels): self.tfidf.fit(texts); X=self.tfidf.transform(texts); self.clf.fit(X,labels); return self
    def predict_proba(self,texts): return self.clf.predict_proba(self.tfidf.transform(texts))
    def predict(self,texts): return self.clf.predict(self.tfidf.transform(texts))
    def top_terms(self,text,label,limit=5):
        row=self.tfidf.transform([text])[0]; names=self.tfidf.get_feature_names_out(); ci=self.clf.classes_.index(label)
        ranked=sorted(((v*self.clf.coef_[ci][j],names[j]) for j,v in row.items()), reverse=True)
        return [term for score,term in ranked if score>0][:limit] or [term for score,term in ranked[:limit]]

def save_model(model,path):
    with open(path,'wb') as f: pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
def load_model(path):
    with open(path,'rb') as f: return pickle.load(f)
