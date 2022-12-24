import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json

st.markdown("""
<style>
.dataf {
    font-size:50px !important;
    text-align: center;
    color: black
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="dataf">DATAFRAME</p>', unsafe_allow_html=True)



def add_bg_from_url():

    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url("https://media.istockphoto.com/id/1278709873/photo/brown-recycled-paper-crumpled-texture-background-cream-old-vintage-page-or-grunge-vignette.jpg?b=1&s=170667a&w=0&k=20&c=NqKmm_gkRwJAqpTbiiqv3TwfWjq9ymOwUDwfG2ck9no=");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )




add_bg_from_url() 


df=pd.read_csv('news_articles.csv')
df=df.head()
st.dataframe(data=df, width=None, height=None, use_container_width=False)





# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# class_label = [0, 1]
# df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
# sns.heatmap(df_cm, annot=True, fmt='d')
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()
