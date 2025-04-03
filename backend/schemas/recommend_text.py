from pydantic import BaseModel

class RecommendTextRequest(BaseModel):
    text: str