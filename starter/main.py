# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient
"""
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list] 
    item_id: int
"""
app = FastAPI()

@app.get("/")
async def greetings():
    return "hello"

"""
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item
"""

client = TestClient(app)
r = client.get("/")
assert r.status_code == 200
print(r.json())
