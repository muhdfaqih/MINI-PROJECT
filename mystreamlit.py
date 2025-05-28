import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

#st.image(r'C:\Users\ACER\welcome\Desktop\Header.jpg') #lain location
st.image('Header.jpg') #location sama dgn streamlit

st.date_input("Select a date")

st.title("""Welcome
This is Old Trafford""")

#upload data
#upload_file = st.file_uploader("Please upload here:", type = 'csv')


#df = pd.read_csv(r"C:\Users\welcome\Desktop\BSMS1306\streamlit\Tips.csv")
df = pd.read_csv("Tips.csv")
#df = pd.read_csv(upload_file)

#show data
st.subheader("Raw Data")
st.write(df)

#histogram
st.subheader("Histogram")
column = st.selectbox("Choose a column",df.columns)
#fig, ax = plt.subplots(figsize = (10,6))
#df[column].plot(kind = 'hist', ax =ax)
#st.pyplot(fig)
fig = px.histogram(df, x=column)
fig.update_traces( marker = {"color":"purple", "line":{"color":"black","width":2}})
st.plotly_chart(fig)

#Scatter chart
st.subheader("Scatter Chart")
x_column = st.selectbox("Choose x-axis column",df.columns)
y_column = st.selectbox("Choose y-axis column",df.columns)
fig, ax = plt.subplots(figsize = (10,6))
df.plot(kind = 'scatter', x=x_column, y=y_column, ax =ax)
st.pyplot(fig)

#fig = px.scatter(df, x=x_column, y = y_column,color ='sex' , color_discrete_sequence= ['yellow', 'red'])
#st.plotly_chart(fig)

