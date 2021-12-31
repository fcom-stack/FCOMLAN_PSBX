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
import datetime
import numpy as np


import streamlit as st
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud 
from matplotlib import pyplot as plt
from ipywidgets import widgets


import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


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
    data['date'] = data['starttime'].dt.date
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
    data = df.drop_duplicates(subset=['artistname', 'trackname'], keep="first")
    inter_df = data.apply(lambda row : get_features(row.artistname, row.trackname),axis=1)
    return reduce(lambda df1,df2 : df1.append(df2), inter_df)

def get_list_genres(df) :
    liste = df.genres.apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']]
    return liste

def extract_myartist(df):
    my_artist = list(set(df.artistname))
    my_artist.remove("")
    return pd.DataFrame(my_artist)

def get_quarter_df(df,quarter):
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
    
def get_month_df(df,month):
    return df[df.month == int(month)]  
    
def get_total_duration_by_(df,column):
    new_df = df.groupby(column).sum()['minplayed'].reset_index().sort_values('minplayed', ascending=False).rename(columns={'minplayed': 'duration_min'})
    return new_df

def get_moy_duration_by_(df,column):
    new_df = df.groupby(column).mean()['minplayed'].reset_index().sort_values('minplayed', ascending=False).rename(columns={'minplayed': 'duration_min'})
    return new_df

def get_tracks_by_date(df,date):
    new_df = df[df.date == date]
    new_df = new_df.groupby(['trackname','artistname']).sum()['minplayed'].reset_index().sort_values('minplayed', ascending=False).rename(columns={'minplayed': 'duration_min'})    
    return new_df

def get_top5_by_(df,group,column):
    if group =='':
         return df.groupby(column).agg({'minplayed':'sum','msplayed' : 'count'}).reset_index().sort_values(['minplayed','msplayed'], ascending=False).rename(columns={'minplayed': 'duration_min','msplayed':'total_count'}).head(5)    
                    
    else :    
        subgroup = df[group].drop_duplicates()
        new_df = subgroup.apply(lambda var : df[df[group] == var].groupby(column).agg({'minplayed':'sum','msplayed' : 'count'}).reset_index().sort_values(['minplayed','msplayed'], ascending=False).rename(columns={'minplayed': 'duration_min','msplayed':'total_count'}).head(5))    
        return reduce(lambda df1,df2 : df1.append(df2), new_df).sort_values([group,'duration_min'], ascending=(True,False))

def get_list_features_by_track(df,features):
    return df.merge(features, how='left', on='trackname')



#--------------------------------- ---------------------------------  ---------------------------------
#---------------------------------              CHARTS
#--------------------------------- ---------------------------------  ---------------------------------

def wordcloud(df):
    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(' '.join(df))

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()    
    return st.pyplot()

def time_series(df,x,y,title,x_label,y_label,x_max,y_max):
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(df[x]), y=list(df[y])))
    
    # endpoints
    fig.add_trace(go.Scatter(x=[x_max],y=[y_max],mode='markers',showlegend=True,marker=dict(color='red', size=12)))

    # Set title
    fig.update_layout(title_text=title)

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,label="1m",step="month", stepmode="backward"),
                    dict(count=6,label="6m",step="month",stepmode="backward"),
                    dict(count=1,label="YTD",step="year",stepmode="todate"),
                    dict(count=1,label="1y",step="year",stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return st.plotly_chart(fig)

def histogram(df,x,y,title,x_label,y_label) :
    fig = px.bar(df, x=x, y=y,title=f''+title, labels={'x':x_label, 'y':y_label})
    fig.update_xaxes(rangeslider_visible=True)
    return st.plotly_chart(fig)

def sunburst(df,child,parent,value):
    fig = px.sunburst(df, path=[child,parent], values=value, color=parent)
    return st.plotly_chart(fig)


def piechart(df,values,names,title):
    fig = px.pie(df, values=values, names=names, title=title)
    return st.plotly_chart(fig)

def barchart(df,x,y,text,title):
    fig = px.bar(df, x=x, y=y,color=x,text=text, title=title, width=700)
    return st.plotly_chart(fig)

def heatmap(df,x,y,z):   
    fig = px.density_heatmap(df, x=x, y=y, z=z)
    return st.plotly_chart(fig)

def animated_bar(df,x,y,animation_frame):
    fig = px.bar(df, x=x, y=y, color=y,animation_frame=animation_frame,width=900,height=600)
    return st.plotly_chart(fig)

def mse_chart(df,x,y):
    fig = px.scatter(df,  x=x, y=y, opacity=0.65,trendline='ols', trendline_color_override='darkblue')
    return st.plotly_chart(fig)

def parralel(df):
    fig = px.parallel_coordinates(df, labels={"danceability": "danceability",
                    "energy": "energy", "valence": "valence",
                    "tempo": "tempo", "loudness": "loudness", },
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                 color_continuous_midpoint=2)
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

# last date
max_date = my_data['date'].max()

# first date
min_date = my_data['date'].min()

duration_by_day = get_total_duration_by_(my_data,'date').sort_values('date', ascending=False)


max_duration_by_day = duration_by_day["duration_min"].max()

day_max_duration = duration_by_day[duration_by_day["duration_min"] == max_duration_by_day]['date']


moy_duration_by_month_day = get_moy_duration_by_(my_data,['month','day'])

moy_duration_by_dayofweek_hour = get_moy_duration_by_(my_data,['dayofweek','hour'])



top5_track = get_top5_by_ (my_data,'','trackname')

top5_track_by_quarter = get_top5_by_(my_data,'quarter',['quarter','trackname'])

top5_track_by_month = get_top5_by_(my_data,'month',['month','trackname'])



################# On spotify api ########################

#features = create_features_df(my_data)
#features.to_csv("features.csv")

features = pd.read_csv("data/features.csv")

code = '''def get_features(artist,track):  
    results = sp.search(q='artist:' + artist, type='artist')
    items = results['artists']['items']    
    return items '''

liste_genres = get_list_genres(features)

total_artist = my_data['artistname'].nunique()


################# spotify api / personal data ########################


list_features_for_top5_track  = get_list_features_by_track(top5_track,features)

list_features_for_top5_track_by_quarter  = get_list_features_by_track(top5_track_by_quarter,features)

list_features_for_top5_track_by_month = get_list_features_by_track(top5_track_by_month,features)


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
     "III- Analysis",
     "IV- Prediction")
)

    #---------------------------------           Part1                  -------------------------------

if select == "I- Introduction":

    st.header("1- Introduction")
    
    par = "Welcome to my wonderful musical universe :smile:. Let's discover together what's behind the data stored and analyzed by Spotify. Let's go !!!! "
    st.write(par)
    
    # Create and generate a word cloud image:
    wordcloud(liste_genres.value)
    
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
    
    st.header("II- Quick look on the data")  
    
    select = st.sidebar.selectbox(
    "SELECT A DATASOURCE👇:",
    ("1- Personal Spotify Data",
     "2- API Spotify")
    )
    
    if select == "1- Personal Spotify Data":
        
        col1, col2,col3= st.columns([1,1,2])
        col1.metric(label="No. of artists", value=total_artist)
        col2.metric(label="No. of tracks", value=total_songs)
        col3.metric(label="Total duration", value=str(total_duration)[0:16])
        col1,col2= st.columns(2)
        col1.metric(label="From", value=str(max_date))
        col2.metric(label="To", value=str(min_date))             
        
        st.markdown(" \n\n ")
        
        with st.expander("You can click here to see the raw data first 👉"):
            st.dataframe(my_data.drop('duration',axis = 1))
                
    if select == "2- API Spotify":
        
        col1, col2= st.columns(2)
        col1.metric(label="No. of artists", value=total_artist)
        col2.metric(label="No. of tracks", value=total_songs)
                
        st.write("To write the query to the Spotify api to build the following dataframe, use as a key the list of items you listened to coupled with the list of artists related to them")   
        
        st.code(code, language='python')
        
        st.markdown(" \n\n ")
        
        with st.expander("You can click here to see the raw data first 👉"):
            st.dataframe(features)
    
 #---------------------------------           Part3                  -------------------------------

if select == "III- Analysis":
    st.header("III- Analysis")   
 
    # Allow use to choose
    #quarter = st.sidebar.radio("Visualize by quarter",('All','Q1', 'Q2', 'Q3','Q4'))
    #df_quarter = get_quarter(my_data,quarter)
    
    ################# Total time spent on Spotify time series ########################
        
    time_series(duration_by_day,'date','duration_min','Total time spent on Spotify :'+str(total_duration)[0:16],'Date','Minutes spent',day_max_duration.values[0],max_duration_by_day) 
    
    st.markdown("### The highest listening time was hit on "+str(day_max_duration.values[0])+". Let's see the tracks that kept me there that day.")
    
    st.write("\n\n")
    
    ################# Track listen on each day ########################
    

    start_date = st.date_input('Select a date', day_max_duration.values[0],min_value=min_date, max_value=max_date)
        
    sunburst(get_tracks_by_date(my_data,start_date).head(10),'artistname','trackname','duration_min')
            

    ################# Habits by something  ########################
    
                
    st.markdown("### What about my daily listening habits ?")
    
    group = st.selectbox("Display by :",('Month and day','Weekday and hour'))
    
    if group == 'Month and day':
        heatmap(moy_duration_by_month_day,'month','day','duration_min')
    else :
        heatmap(moy_duration_by_dayofweek_hour,'dayofweek','hour','duration_min')
    
    
    ################# Top artist /track /genre  ########################
            
    st.markdown("### Let's take a look at my musical interests now")   
    
    group = st.selectbox("Display :",('Overall in 2021','By quarter','By month'))
    
    if group == 'Overall in 2021':
        
        with st.expander("Dropdown if you want to listen my favorite song of the year 👉"):
            st.video("https://www.youtube.com/watch?v=pRweltAO-zg")
        
        all_features = get_list_features_by_track(top5_track,features)
        barchart(all_features,'duration_min','trackname','artistname','Top 5 songs / artists')
        
        wordcloud(get_list_genres(all_features).value)
        
        parralel(list_features_for_top5_track)
              
        
    elif group == 'By quarter':
        
        #animated_bar(top5_track_by_quarter,"duration_min","trackname","quarter")
        
        q = st.select_slider('Select a quarter',options=['Q1', 'Q2', 'Q3','Q4'])
        data = get_quarter_df(top5_track_by_quarter,q)
        all_features = get_list_features_by_track(data,features)
        
        barchart(all_features,'duration_min','trackname','artistname','Top 5 songs / artists by quarter')
        
        wordcloud(get_list_genres(all_features).value)
        
    else: 
        m = st.select_slider('Select a month',options=['1', '2', '3','4','5','6','7','8','9','10','11','12'])
        
        data = get_month_df(top5_track_by_month,m)
        
        
        barchart(all_features,'duration_min','trackname','artistname','Top 5 songs / artists by month')
        
        wordcloud(get_list_genres(all_features).value)
            

#---------------------------------           Part4                  -------------------------------

if select == "IV- Prediction":
    st.header("IV- Prediction")         
    
    st.markdown("### Now that we know what my listening habits are, what can we imagine for the next year in terms of daily listening time ?")
    
    
    cars_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/imports-85.csv')

    # Build parcats dimensions
    categorical_dimensions = ['body-style', 'drive-wheels', 'fuel-type'];

    dimensions = [dict(values=cars_df[label], label=label) for label in categorical_dimensions]

    # Build colorscale
    color = np.zeros(len(cars_df), dtype='uint8')
    colorscale = [[0, 'gray'], [1, 'firebrick']]

    # Build figure as FigureWidget
    fig = go.FigureWidget(
        data=[go.Scatter(x=cars_df.horsepower, y=cars_df['highway-mpg'],
        marker={'color': 'gray'}, mode='markers', selected={'marker': {'color': 'firebrick'}},
        unselected={'marker': {'opacity': 0.3}}), go.Parcats(
            domain={'y': [0, 0.4]}, dimensions=dimensions,
            line={'colorscale': colorscale, 'cmin': 0,
                  'cmax': 1, 'color': color, 'shape': 'hspline'})
        ])

    fig.update_layout(
            height=800, xaxis={'title': 'Horsepower'},
            yaxis={'title': 'MPG', 'domain': [0.6, 1]},
            dragmode='lasso', hovermode='closest')

    # Update color callback
    def update_color(trace, points, state):
        # Update scatter selection
        fig.data[0].selectedpoints = points.point_inds

        # Update parcats colors
        new_color = np.zeros(len(cars_df), dtype='uint8')
        new_color[points.point_inds] = 1
        fig.data[1].line.color = new_color

    # Register callback on scatter selection...
    fig.data[0].on_selection(update_color)
    # and parcats click
    fig.data[1].on_click(update_color)

    st.plotly_chart(fig)
    
    #mse_chart(duration_by_day,'date','duration_min')
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
