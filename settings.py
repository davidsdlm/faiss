from pydantic_settings import BaseSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class Settings(BaseSettings):
    DB_URL: str


settings = Settings()

engine = create_engine(settings.DB_URL)
Session = sessionmaker(engine)
