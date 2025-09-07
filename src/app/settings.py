from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import ClassVar


class Settings(BaseSettings):
    load_dotenv(override=True)
    
    # openai client = OpenAI(
    client: ClassVar[OpenAI] = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    )
    brain_model: str = "openai/gpt-oss-20b:cerebras"


    

