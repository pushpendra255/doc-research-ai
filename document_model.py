from pydantic import BaseModel

class Document(BaseModel):
    doc_id: str
    content: str
