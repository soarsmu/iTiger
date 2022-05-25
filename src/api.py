from fastapi import Depends, FastAPI
from pydantic import BaseModel
from title_generator.model import get_model

app = FastAPI()

class IssueTitleResponse(BaseModel):
    title: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict", response_model=IssueTitleResponse)
async def predict(text: str, model = Depends(get_model)):
    return IssueTitleResponse(
        title=model.predict(text)
    )

