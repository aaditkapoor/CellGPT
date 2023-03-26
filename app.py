import streamlit as st
import pandas as pd
import os
from streamlit_imagegrid import streamlit_imagegrid
from PIL import Image
import requests

def prepare_input(image, text):

    image = Image.open(requests.get(image, stream=True).raw)
    text = text
    encoding = processor(image, text, return_tensors="pt")
    return encoding


def get_prediction(encoding):
    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])


from transformers import AutoProcessor, ViltForVisualQuestionAnswering
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForVisualQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

st.title("CellGPT: Ask questions to biologist")


query = st.text_input("What would you like to know?")

urls = [
        {
          "width": 400,
          "height": 400,
          "src": "https://cildata.crbs.ucsd.edu/media/thumbnail_display/54692/54692_thumbnailx512.jpg"
        },
        {
          "width": 400,
          "height": 400,
          "src": "http://cellimagelibrary.org/pic/thumbnail_display/25379/nucleus.jpg"
        },
        {
          "width": 1000,
          "height": 666,
          "src": "http://cellimagelibrary.org/pic/thumbnail_display/11406/mitochondrion.jpg"
        },
         {
          "width": 1000,
          "height": 666,
          "src": "http://cellimagelibrary.org/pic/thumbnail_display/48102/mammary+epithelial+cell.jpg"
        },

      ]

return_value = streamlit_imagegrid(urls=urls,height=1000)
if query and return_value:
    encoding = prepare_input(image=return_value['src'], text=query)
    st.info("Query: ", query)
    pred = get_prediction(encoding)
    st.write(pred)
else:
    st.error("No query!")
