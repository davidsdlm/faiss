from fastapi import FastAPI
import uvicorn

from app.v1.routers import router as v1_routers

app = FastAPI()

app.mount("/v1/", v1_routers)

if __name__ == "__main__":
    uvicorn.run(app, port=8002)
