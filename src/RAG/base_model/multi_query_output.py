from pydantic import BaseModel 
from typing import List

class MultiQueryOutput(BaseModel):
    questions: List[str]