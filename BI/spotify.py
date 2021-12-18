import pandas as pd
import json
import datetime
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time 
from functools import reduce
from datetime import timedelta
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


import streamlit as st
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud 
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings("ignore")


#--------------------------------- ---------------------------------  ---------------------------------
#---------------------------------              FUNCTIONS
#--------------------------------- ---------------------------------  ---------------------------------

def load_myspotify_data():
    data = pd.read_json('data/StreamingHistory.json')
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data['endtime'] = pd.to_datetime(data['endtime'])
    data['duration'] = pd.to_timedelta(data['msplayed'], 'ms')
    data['minplayed'] = data['msplayed'] / 1000 / 60
    data['starttime'] = data['endtime'] - data['duration']
    data['year'] = data['starttime'].dt.year
    data['month'] = data['starttime'].dt.month
    data['week'] = data['starttime'].dt.isocalendar().week
    data['dayofweek'] = data['starttime'].dt.dayofweek
    data['day'] = data['starttime'].dt.day
    data['hour'] = data['starttime'].dt.hour
    data['minute'] = data['starttime'].dt.minute
    data['quarter'] = data['starttime'].dt.quarter
    return data

#spotify api

client_id = 'ef0d92753d4e44658a3a28ce21de6845'
client_secret = '828972dcf8b84b9f8fe76e27cd570c6d'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_features(artist,track):  
    
    columns = ["artistName","followers","genres","popularity","artistType",
               "track","danceability","energy","loudness","speechiness","instrumentalness",
              "liveness","valence","tempo","duration_ms"]
    
    df = pd.DataFrame([[artist,"","","","",track,"","","","","","","","",""]],columns = columns )
    
    results = sp.search(q='artist:' + artist, type='artist')
    items = results['artists']['items']
    
    if len(items) > 0:
        #get artist features
        features = items[0]
        df.followers[0] = features['followers']['total']
        df.genres[0] = features['genres']
        df.popularity[0] = features['popularity']
        df.artistType[0] = features['type'] 
    
    results = sp.search(q='artist:'+artist+' track:'+track,type='track')
    items = results['tracks']['items']   
    
    if len(items) > 0:
        features = sp.audio_features(items[0]['id'])[0]
        if(features is not None):
            df.danceability[0] = features['danceability']
            df.energy[0] = features['energy']
            df.loudness[0] = features['loudness']
            df.speechiness[0] = features['speechiness']
            df.instrumentalness[0] = features['instrumentalness']
            df.liveness[0] = features['liveness']
            df.valence[0] = features['valence']
            df.tempo[0] = features['tempo']
            df.duration_ms[0] = features['duration_ms']                

    return df

def create_features_df(df): 
    data = df.drop_duplicates(subset=['artistname', 'trackname'], keep=False)
    inter_df = data.apply(lambda row : get_features(row.artistname, row.trackname),axis=1)
    return reduce(lambda df1,df2 : df1.append(df2), inter_df)

def get_list_genres(df) :
    liste = df.genres.apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']]
    return liste

def extract_myartist(df):
    my_artist = list(set(df.artistname))
    my_artist.remove("")
    return pd.DataFrame(my_artist)

def get_quarter(df,quarter):
    if quarter == 'Q1':
        return df[df.quarter == 1] 
    elif quarter == 'Q2':
        return df[df.quarter == 2] 
    elif quarter == 'Q3':
        return df[df.quarter == 3]    
    elif quarter == 'Q4':
        return df[df.quarter == 4]
    else :
        return df      
    
def get_total_duration_by_(df,column):
    new_df = df.groupby(column).sum()['minplayed'].reset_index().sort_values('minplayed', ascending=False).rename(columns={'minplayed': 'amount'})
    return new_df

def get_moy_duration_by_(df,column):
    new_df = df.groupby(column).mean()['minplayed'].reset_index().sort_values('minplayed', ascending=False).rename(columns={'minplayed': 'amount'})
    return new_df

def get_top10_(df,column):
    new_df = df.groupby(column).sum()['minplayed'].reset_index().sort_values('minplayed', ascending=False).rename(columns={'minplayed': 'amount'}).head(10)

#--------------------------------- ---------------------------------  ---------------------------------
#---------------------------------              CHARTS
#--------------------------------- ---------------------------------  ---------------------------------

def histogram(df,x,y,title,x_label,y_label) :
    fig = px.histogram(df, x=x, y=y,title=f''+title, labels={'x':x_label, 'y':y_label},nbins = 100)
    fig.update_xaxes(rangeslider_visible=True)
    return st.plotly_chart(fig)

def piechart(df,values,names,title):
    fig = px.pie(df, values=values, names=names, title=title)
    return st.plotly_chart(fig)

#--------------------------------- ---------------------------------  ---------------------------------
#---------------------------------              DATAFRAMES
#--------------------------------- ---------------------------------  ---------------------------------

################# On my personal data ########################

my_data = load_myspotify_data()

# Number of gathered songs
total_artist = my_data['artistname'].nunique()

# Number of gathered songs
total_songs = my_data['trackname'].nunique()

# Total duration 
total_duration = my_data['duration'].sum()


################# On spotify api ########################

#features = create_features_df(my_data)
#features.to_csv("features.csv")

features = pd.read_csv("data/features.csv")

liste_genres = get_list_genres(features)

total_artist = my_data['artistname'].nunique()

#--------------------------------- ---------------------------------  ---------------------------------
#---------------------------------                TITLE
#--------------------------------- ---------------------------------  ---------------------------------

left,right = st.columns([3, 1])

with left:
    st.title("MY 2021 YEAR ON SPOTIFY")
    
with right: 
    spotify_logo = Image.open("image/images.jpeg")
    st.image(spotify_logo,width=100)

#--------------------------------- ---------------------------------  ---------------------------------
#---------------------------------               SIDEBAR
#--------------------------------- ---------------------------------  ---------------------------------
       
st.sidebar.header("Florine COMLAN")   

streamlit = "https://docs.streamlit.io"

st.sidebar.write('I built this website in order to show you how you can create a simple visualization project with the [streamlit package](%s). \n\n Find me on my networks to follow my adventures :wink:.'%streamlit)

linkedin = "https://www.linkedin.com/in/florine-comlan/" 
github = "https://github.com/fcom-stack"
       
st.sidebar.image(Image.open("image/linkedin.png"),width=30)  
st.sidebar.write(" [My LinkedIn](%s)" % linkedin)
st.sidebar.image(Image.open("image/github.png"),width=30) 
st.sidebar.write(" [My Github](%s) \n\n" % github)  

    
select = st.sidebar.selectbox(
    "SELECT A FEATURE 👇:",
    ("I- Introduction",
     "II- Quick look on the data",
     "III- Analysis")
)

    #---------------------------------           Part1                  -------------------------------

if select == "I- Introduction":

    st.header("1- Introduction")
    
    par = "Welcome to my wonderful musical universe :smile:. Let's discover together what's behind the data stored and analyzed by Spotify. Let's go !!!! "
    st.write(par)
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(' '.join(liste_genres.value))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
    
    
    st.header("2- Data sources")
    
    link1 = "https://support.spotify.com/us/article/data-rights-and-privacy-settings/"
    link2 = "https://medium.com/@rafaelnduarte/how-to-retrieve-data-from-spotify-110c859ab304"
    
    par = 'To start, it is important to describe the data that will be the subject of our analysis in this project.\n Before the end of each year, Spotify submits to its users a list of the top sounds that have marked the past year. This is how I came up with the idea of analyzing my music data to find out more about my habits.\n\nTo perform this personal data analysis, I used a summary of the list of items (e.g. songs, videos and podcasts) that I have listened to or watched in the past year (December 2020 to December 2021). To also access your Spotify data, simply complete the instructions in the "Can I download my personal data?" section by following this link:'    
    
    st.write(par)
    st.write("[Download Personal Spotify Data Tutorial](%s)" % link1)
    
    par = 'On the other hand I also used the official Spotify api to complete the list of data I had access to through my personal data. I collected information about the genres of the artists for example I had to listen to in order to make a cross analysis. To access the data provided by the Spotify api, follow the tutorial below:'
    
    st.write(par)
    st.write("[Using Spotify API Tutorial](%s)" % link2)
    
    
    #---------------------------------           Part2                  -------------------------------

if select == "II- Quick look on the data":
    
    st.header("3- Quick look on the data")  
    
    select = st.sidebar.selectbox(
    "SELECT A DATASOURCE👇:",
    ("1- Personal Spotify Data",
     "2- API Spotify")
    )
    
    if select == "1- Personal Spotify Data":
        
        row1,row2 = st.columns((1,3))

        with row1:
            title = ":microphone::minidisc: Total artist"
            
            str_artists = ":microphone::minidisc: " + str(total_artist) + " \n artists/poadcast"
            st.markdown(str_artists)
        
        with row2:
            str_songs = ":musical_note: " + str(total_songs) + " \n songs"
            st.markdown(str_songs)
            
        with row3:
            str_duration = ":hourglass: " + str(total_duration) + " \n total listening time"
            st.markdown(str_duration)
            
        spacer1, row1, spacer2 = st.columns((.2, 7.1, .2))
        
        with row1:
            st.markdown("")
            with st.expander("You can click here to see the raw data first 👉"):
                
                st.dataframe(my_data.drop('duration',axis = 1))
                
    if select == "2- API Spotify":
        
        st.write("write")
    
 #---------------------------------           Part3                  -------------------------------

if select == "III- Analysis":
    st.header("4- Analysis")   
 
    # Allow use to choose
    quarter = st.sidebar.radio("Visualize by quarter",('All','Q1', 'Q2', 'Q3','Q4'))
    df_quarter = get_quarter(my_data,quarter)
    
    ################# Total time spent on Spotify histogram ########################
    
    histogram(df_quarter,'starttime','minplayed','Total time spent on Spotify','Date','Minutes spent')
    
    
    
    
    ################# Artist listen the most ########################
    
    df_top_artist = get_top10_artist(df_quarter)
    
    piechart(df_top_artist,'amount','artistname','Top 10 artists listen the most')
    
    
    fig = px.sunburst(df_quarter, path=['artistname', 'trackname'], values='minplayed', color='artistname')
    st.plotly_chart(fig)
    
    
    
        
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
