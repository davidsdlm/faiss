import numpy as np
import sys

from sqlalchemy import select
import faiss

from settings import Session
from app import models

with Session() as session:
    companies = session.execute(select(models.News.id.label("id"), models.News.embedding)).all()
    ids: np.array(int) = np.array([company[0] for company in companies])
    embeddings: np.array(np.array(float)) = np.array([np.array(company[1]) for company in companies])

dimension = len(embeddings[0])

_index = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(_index)
index.add_with_ids(np.array(embeddings), np.array(ids))
