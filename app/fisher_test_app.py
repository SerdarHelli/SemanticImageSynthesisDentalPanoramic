import streamlit as st
import os
from PIL import Image
import pandas as pd
from gspread_pandas import Spread,Client
from google.oauth2 import service_account
import random

from time import time



absolute_path = os.path.dirname(__file__)
relative_path = "dir"

dirname = os.path.join(absolute_path, relative_path)

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
json={
  "type": "...",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "...",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}







if 'client' not in st.session_state:
    credentials = service_account.Credentials.from_service_account_info(
                    json, scopes = scope)
    st.session_state.client = Client(scope=scope,creds=credentials)
  
if 'spread' not in st.session_state:
    spreadsheetname = "SavedResults"
    st.session_state.spread = Spread(spreadsheetname,client = st.session_state.client)

if "sh" not in st.session_state:
    spreadsheetname = "SavedResults"
    st.session_state.sh = st.session_state.client.open(spreadsheetname)


if "worksheet_list" not in st.session_state:
    st.session_state.worksheet_list = st.session_state.sh.worksheets()




def worksheet_names():
    sheet_names = []   
    for sheet in st.session_state.worksheet_list:
        sheet_names.append(sheet.title)  
    return sheet_names
    
# Get the sheet as dataframe
@st.cache
def load_the_spreadsheet(spreadsheetname):
    worksheet = st.session_state.sh.worksheet(spreadsheetname)
    df = pd.DataFrame(worksheet.get_all_records())
    return df

# Update to Sheet
@st.cache
def update_the_spreadsheet(spreadsheetname,dataframe):
    col = [ 'Image','FakeOrNot','Elapsed Time(sec)','Elapsed Time(min)']
    st.session_state.spread.df_to_sheet(dataframe[col],sheet = spreadsheetname,index = False)

if 'dir_path' not in st.session_state:
    st.session_state.dir_path = ''
    
st.session_state.dir_path=dirname

    
if 'count' not in st.session_state:
        st.session_state.count = 0
        
if 'results' not in st.session_state:
        what_sheets = worksheet_names()
        st.session_state.results = load_the_spreadsheet(what_sheets[-1])

if 'timer' not in st.session_state:
    st.session_state.timer = 0
    

if 'dirs' not in st.session_state:
    st.session_state.dirs=  [f for f in os.listdir(st.session_state.dir_path) if os.path.isfile(os.path.join(st.session_state.dir_path, f)) and  f.endswith(".png") or f.endswith(".jpg") ]
    random.shuffle(st.session_state.dirs)



def save():
    what_sheets = worksheet_names()
    update_the_spreadsheet(what_sheets[-1],st.session_state.results)


if 'done' not in st.session_state:
    st.session_state.done =0

def endlog():
    end = time()
    st.session_state.timer=end-start

def getTime():
    return float(st.session_state.timer)
    
def increment_counter_genareted():
    if not hasattr(st.session_state, 'dirs'):
        st.error("Cache is empty . Reruning... !")
    else:
        endlog()
        data={'Image': st.session_state.dirs[get_counter()], 'FakeOrNot': 0,'Elapsed Time(sec)':str(round(getTime(),5))+" sec",'Elapsed Time(min)':str(round((getTime()/60),5))+" min"}
        st.session_state.results=st.session_state.results.append(data,ignore_index=True)
        st.session_state.count += 1

def get_counter():
    if st.session_state.count==None:
        return None
    return int(st.session_state.count)

def increment_counter_real():
    if not hasattr(st.session_state, 'dirs'):
        st.error("Cache is empty .Reruning... !")
    else:
        endlog()
        data={'Image': st.session_state.dirs[get_counter()], 'FakeOrNot': 1,'Elapsed Time(sec)':str(round(getTime(),5))+" sec",'Elapsed Time(min)':str(round((getTime()/60),5))+" min"}
        print(round(getTime(),5))
        st.session_state.results=st.session_state.results.append(data,ignore_index=True)
        st.session_state.count += 1
    
def get_resultslist():
    return st.session_state.results["Image"].values

def get_invalid_counter():
    for i in range(st.session_state.count,len(st.session_state.dirs)):
        if not st.session_state.dirs[i] in get_resultslist():
            return i 
    return None



   
mystyle = '''
<style>
    .appview-container  {
        text-align: center;
    
    }
</style>
'''

st.markdown(mystyle, unsafe_allow_html=True)    
st.markdown("<h1 style='  color: #ff4b4b;'>Turing Test : Semantic Image Synthesis Dental Panaromic</h1>", unsafe_allow_html=True)

if not hasattr(st.session_state, 'dirs'):

    st.error("Cache is empty .Reruning... !")


else:
      
    if get_counter()!=None:
        if len(st.session_state.dirs)>(get_counter()):
        
            if st.session_state.dirs[get_counter()] in get_resultslist():
                st.session_state.count=get_invalid_counter()
            if st.session_state.count!=None:
                start = time()    
                image_path=os.path.join(st.session_state.dir_path, st.session_state.dirs[get_counter()])
                image=Image.open(image_path)
                coli1, coli2 = st.columns([2,10])

                with coli2:
                    st.image(image)
                col1, col2,col3 = st.columns([1,5,5])
                with col2:
                    generated_button=st.button("Genareted",on_click=increment_counter_genareted)  
                with col3:
                    real_button=st.button("Real",on_click=increment_counter_real)

            else:
                st.markdown("***")
                st.success('Done ! ❤  Please contact us !')
                st.balloons()

        else:
            st.markdown("***")
            st.success('Done ! ❤  Please contact us !')
            st.balloons()

    else:
        st.markdown("***")
        st.success('Done ! ❤  Please contact us !')
        st.balloons()

    st.markdown("***")


    colT1,colT2 = st.columns([1,5])
    with colT2:

        if st.button("Please press to save your progress",on_click=save):
            st.info("Your progress has been saved!")
        st.empty()
        st.caption("Don't forget to save your progress. So you can contiune where  you left...")



footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: #ff4b4b;
text-align: center;
}
</style>
<div class="footer">
    <div class="Row">
            <div class="Column">    
                <img  src="https://github.com/ImagingYeditepe/ImagingYeditepe.github.io/raw/main/img/imaging_logo.png"  width="65" alt="Logo" >
            </div>
            <div class="Column"> 
                Developed with ❤ by <a style='display: block; text-align: center; color: #ff4b4b;' href="https://imagingyeditepe.github.io/" target="_blank">Yeditepe Imaging </a>
            </div>
    </div>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)