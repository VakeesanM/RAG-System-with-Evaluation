from typing import Union
from pydantic import BaseModel 

class GenOutput(BaseModel):
    output: Union[str, int]