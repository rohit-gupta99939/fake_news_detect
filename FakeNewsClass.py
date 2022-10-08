import string
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class NewsClass(BaseModel):
    tit: str
    text:str