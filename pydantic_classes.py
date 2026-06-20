from pydantic import BaseModel,Field,field_validator
from fastapi import HTTPException
from typing import Annotated,Literal

class VideoRequest(BaseModel):
    url: str
class Query(BaseModel):
    query: str