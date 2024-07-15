import structlog
from fastapi import FastAPI
import numpy as np

from app.v1 import schemas
from app.v1 import vector_engine
from app.v1 import tf_idf_engine

router = FastAPI()
logger = structlog.getLogger("router")


@router.post("/vector_search/")
async def search_by_vector(data: schemas.SearchVectorRequest):
    result = vector_engine.index.search(np.array([data.embedding]), data.n)
    return schemas.SearchResponse(company_ids=result[1][0])


@router.post("/token_search/")
async def search_by_token(data: schemas.SearchTokensRequest):
    text = " ".join(data.tokens)
    embedding = tf_idf_engine.vectorizer.transform([text]).toarray()
    result = tf_idf_engine.index.search(embedding, data.n)
    return schemas.SearchResponse(company_ids=result[1][0])
