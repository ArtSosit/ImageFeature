import cv2
import numpy as np
from typing import Union 
from fastapi import FastAPI 
from pydantic import BaseModel 
import base64

app = FastAPI()

class Item(BaseModel):
    image_base64: str

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) 


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None): 
    return {"item_id": item_id, "q": q}

@app.post("/api/genhog")
def Image_Features(data: Item):
    
    img_gray = readb64(data.image_base64)

    img_new = cv2.resize(img_gray, (128,128), cv2.INTER_AREA) 
    win_size = img_new.shape

    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins) 
     hog_descriptor = hog.compute(img_new) 
    return {"vector": hog_descriptor.tolist()}
