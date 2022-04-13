import streamlit as st
from PIL import Image
import glob
import time
import requests
import json
import os
import numpy as np
import pandas as pd
import pydeck as pdk
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import gdown
from torchvision import transforms
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

import io
import base64
import urllib.request
import PyPDF4
import streamlit.components.v1 as components

#from torch.utils.model_zoo import _download_url_to_file

# st.set_page_config(layout="wide")

image_list = []
cap_list=[]
for filename in glob.glob('images/*.jpg'):
    im=Image.open(filename)
    image_list.append(im)
    cap_list.append(filename[13:27])

l=len(image_list)

#st.title('Cyclone Vision Portal')
colT1,colT2 = st.columns([2,8])
with colT2:
    st.title('Cyclone Vision Portal')
#st.title("<h1 style='text-align: center;'>Cyclone Vision Portal</h1>", unsafe_allow_html=True)

im_slot=st.empty()

t = st.slider("INSAT-3D IR visualization", 0,l-1)

im_slot.image(image_list[t], caption=cap_list[t])

if st.button('PLAY'):
    for x in range(l):
        time.sleep(1)
        im_slot.image(image_list[x], caption=cap_list[x])

components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
st.header('Get Elevation for Your Location')
loc_button = Button(label="Click for Your Location")
loc_button.js_on_event("button_click", CustomJS(code="""
    navigator.geolocation.getCurrentPosition(
        (loc) => {
            document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
        }
    )
    """))
latlondata = streamlit_bokeh_events(
    loc_button,
    events="GET_LOCATION",
    key="get_location",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)
if(latlondata):
    lat1=latlondata['GET_LOCATION']['lat']
    lon1=latlondata['GET_LOCATION']['lon']
    lat1x=str(lat1)[0:3]+((len(str(lat1))-3)*'x')
    lon1x=str(lon1)[0:3]+((len(str(lon1))-3)*'x')
    st.write("Latitude: "+lat1x, "Longitude: "+lon1x)
    ele_resp=requests.get("https://api.opentopodata.org/v1/test-dataset?locations="+str(int(lat1))+","+str(int(lon1)))
    ele_json=ele_resp.json()
    st.write("Elevation: "+str(ele_json['results'][0]['elevation']))

st.header("Enter location to get elevation")
locq=st.text_input("Enter Location")

map_loc=st.empty()

df = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [28.65, 77.22],
        columns=['lat', 'lon'])
map_loc.pydeck_chart(pdk.Deck(
map_style='mapbox://styles/mapbox/dark-v10',
initial_view_state=pdk.ViewState(
    latitude=28.65,
    longitude=77.22,
    zoom=12,
    pitch=50,
    ),
    layers=[
    pdk.Layer(
        'HexagonLayer',
        data=df,
        get_position='[lon, lat]',
        radius=100,
        elevation_scale=10,
        elevation_range=[0, 100],
        pickable=True,
        extruded=True,
        ),
    ],
))

if(locq):
    response = requests.get("http://api.openweathermap.org/geo/1.0/direct?q="+locq+"&limit=1&appid=4cc160d504b0aff2ab13074669b93098")
    latlontmp2 = response.json()
    lat2=latlontmp2[0]['lat']
    lon2=latlontmp2[0]['lon']
    st.write("Latitude: "+str(lat2)+", Longitude: "+str(lon2))
    ele_resp2=requests.get("https://api.opentopodata.org/v1/test-dataset?locations="+str(int(lat2))+","+str(int(lon2)))
    ele_json2=ele_resp2.json()
    st.write("Elevation: "+str(ele_json2['results'][0]['elevation']))

    #df=pd.read_csv('ele-data.csv')

    #map_data = pd.DataFrame({'lat': [lat2], 'lon': [lon2]})
    #map(df,lat2,lon2,12)


    df = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [lat2, lon2],
        columns=['lat', 'lon'])

    map_loc.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v10',
    initial_view_state=pdk.ViewState(
        latitude=lat2,
        longitude=lon2,
        zoom=12,
        pitch=50,
        ),
        layers=[
        pdk.Layer(
            'HexagonLayer',
            data=df,
            get_position='[lon, lat]',
            radius=100,
            elevation_scale=10,
            elevation_range=[0, 100],
            pickable=True,
            extruded=True,
            ),
        ],
    ))

components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)



# ----------------------------------------------Inference script -------------------------------------------------------------

#TODO ADD Python inference script
PATH = r"./infer/acd_123_34.jpg"
urla = "https://drive.google.com/uc?id=1XXPduWRnUY582hgfiSddQ2wiz5KR-a0j"
#model_path = r"./model/final_model.ckpt"
if not os.path.exists("model.pt"):
    gdown.download(urla, 'model.pt', quiet = False)
#_download_url_to_file(urla, 'final_model.ckpt', None, True)

# model = PretrainedWindModel.load_from_checkpoint('final_model.ckpt')
# pred = predict_image(sample_image, model)
# st.write(f"Your predicted wind speed is {str(pred)} kts")

#  upload a file in streamlit
st.header("Predict Cyclone Satellite Image Windspeed")
inp = st.file_uploader("Upload The Cyclone Satellite Image", type=["jpg", "png"])
if inp is not None:
    image = Image.open(io.BytesIO(inp.read())).convert("RGB")
    # inp = r"./infer/acd_123_34.jpg" 
    # image = Image.open(inp).convert("RGB")
    test_transforms = transforms.Compose(
            [
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                # All models expect the same normalization mean & std
                # https://pytorch.org/docs/stable/torchvision/models.html
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    image = test_transforms(image)
    image = image.unsqueeze(0)

    scripted_module = torch.jit.load("model.pt")
    output = scripted_module(image)
    output = output.data.squeeze().numpy()
    
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)
        
    c = st.container()
    c.image(inp,caption='Input Cyclone Image')
    c.write(f"Actual wind speed was {inp.name.split('.')[0]} kts")
    c.metric(label="Predicted Wind Speed",value=str(np.round(output,2)) + " kts")
    
# --------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------Alert Pdf---------------------------------------
components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
st.header("Official Alerts")

####################################### This part is for displaying pdf from local using html embeded or ifram
# filename = 'test.pdf' 
# with open(filename, "rb") as f:
#     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     # Embedding PDF in HTML
# #   pdf_display = F'<center><embed src="data:application/pdf;base64,{base64_pdf}" width="400" height="500" type="application/pdf"></center>'
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
#     # Displaying File
#     st.markdown(pdf_display, unsafe_allow_html=True)
####################################### This part is for displaying pdf from local using html embeded or ifram

pdf_link = "https://github.com/inderpreet1390/SIH/raw/main/test.pdf"
pdf_display = F'<center><embed src="https://drive.google.com/viewerng/viewer?embedded=true&url={pdf_link}" width="400" height="400"></center>'
# Displaying File
st.markdown(pdf_display, unsafe_allow_html=True)
    
# -------------------------------------------- Alert PDF --------------------------------------------------
    
# ----------------------------------------------------------------- TEXT Summarizer -------------------------------------------------------
model = 'BART'
_num_beams = 4
_no_repeat_ngram_size = 3
_length_penalty = 1
_min_length = 12
_max_length = 128
_early_stopping = True

# text = st.text_area('Text Input')

def run_model(input_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "BART":
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        input_text = str(input_text)
        input_text = ' '.join(input_text.split())
        input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
        summary_ids = bart_model.generate(input_tokenized,
                                          num_beams=_num_beams,
                                          no_repeat_ngram_size=_no_repeat_ngram_size,
                                          length_penalty=_length_penalty,
                                          min_length=_min_length,
                                          max_length=_max_length,
                                          early_stopping=_early_stopping)

        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        st.write('Summary')
        st.success(output[0])

    else:
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        input_text = str(input_text).replace('\n', '')
        input_text = ' '.join(input_text.split())
        input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
        summary_task = torch.tensor([[21603, 10]]).to(device)
        input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
        summary_ids = t5_model.generate(input_tokenized,
                                        num_beams=_num_beams,
                                        no_repeat_ngram_size=_no_repeat_ngram_size,
                                        length_penalty=_length_penalty,
                                        min_length=_min_length,
                                        max_length=_max_length,
                                        early_stopping=_early_stopping)
        output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        st.write('Summary')
        st.success(output[0])
# convert pdf link to text 
file_pdf = urllib.request.urlopen(pdf_link)
pdfdata = ""
reader = PyPDF4.PdfFileReader(io.BytesIO(file_pdf.read()))

for i in range(reader.numPages):
    page = reader.getPage(i)
    page_text = page.extractText()
    for line in page_text.split('\n'):
        pdfdata += line.strip(' ')
    
if st.button('Summarize'):
    run_model(pdfdata)
# ----------------------------------------------------------------------------------------------------------------------------

