from pydantic import BaseModel

class DrinkRequest(BaseModel):
    beverage: str
    size: str
    milk: str
    whipped_cream: bool