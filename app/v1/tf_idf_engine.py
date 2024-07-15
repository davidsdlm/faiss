import numpy as np
import sys

from sqlalchemy import select
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from settings import Session
from app import models

with Session() as session:
    companies = session.execute(select(models.News.id.label("id"), models.News.tokens)).all()
    ids: np.array(int) = np.array([company[0] for company in companies])
    tokens: np.array(np.array(str)) = np.array([" ".join(company[1]) for company in companies])


vectorizer = TfidfVectorizer()
vectorizer.fit(tokens)
embeddings = vectorizer.transform(tokens).toarray()

dimension = vectorizer.idf_.shape[0]

_index = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(_index)
index.add_with_ids(np.array(embeddings), np.array(ids))
