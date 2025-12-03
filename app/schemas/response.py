from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    probability: float
    class_: int
    xai_image_base64: Optional[str] = None

    class Config:
        fields = {
            'class_': 'class'
        }
