from typing import List, Annotated
from pydantic import BaseModel, Field

class PredictIn(BaseModel):
    input: str

EntityTag = Annotated[str, Field(pattern=r"^[BI]-[A-Z][A-Z_]*$")]

class Span(BaseModel):
    start_index: int
    end_index: int
    entity: EntityTag

PredictOut = List[Span]