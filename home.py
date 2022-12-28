import streamlit as st
import pandas as pd 
import altair as alt
import pickle
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
ps = PorterStemmer()

def root_words(string):
    porter = PorterStemmer()
    
    #  sentence into a list of words
    words = word_tokenize(string)
    
    valid_words = []

    for word in words:
        
        root_word = porter.stem(word)
        
        valid_words.append(root_word)
        
    string = ' '.join(valid_words)

    return string 


tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
mn_model = pickle.load(open('models/mn_model.pkl','rb'))
#adab_model = pickle.load(open('models/adab_model.pkl','rb'))
#bnb_model = pickle.load(open('models/bnb_model.pkl','rb'))
#mn_model = pickle.load(open('models/mn_model.pkl','rb'))


def main():
    # menu = ["Home", "Bio", "Resources"]
    # choice = st.sidebar.selectbox("Menu", menu)
    # if choice == "Home":
    st.subheader("Home")
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
#CSS-HTML Plug in from st.markdoownfor BreakNews
    st.markdown("""
    <style>
    .d {
        font-size:60px !important;
        text-align: center;
    
    
        color: black
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="d"> BREAKING NEWS </p>', unsafe_allow_html=True)
#st.write("Did you know 85% Internet users are tricked by fake news? Are you confident in your ability to detect fake news? Check the authenticity of your news article here ↓ ")
#text_input=st.text_input("Enter The Title Below")

    st.markdown("""
    <style>
    .details {
        font-size:20px !important;
        text-align: left;
    
        font-family:Gothic;
        font-weight: bold;
        color: maroon
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="details">Did you know 85% Internet users are tricked by fake news? Are you confident in your ability to detect fake news? Check the authenticity of your news article here ↓ </p>', unsafe_allow_html=True)
#st.write("Did you know 85% Internet users are tricked by fake news? Are you confident in your ability to detect fake news? Check the authenticity of your news article here ↓ ")
    text_input=st.text_input("Enter The Title Below", key = "<uniquevalueofsomesort>")

#st.markdown(f"",text_input)
    button = st.button("Check")
        # label = {'Fake':0, 'Real':1}
    if button:
        st.success("Original Text")
        st.write(text_input)
        transformed = root_words(text_input)
        vector_input = tfidf.transform([text_input])
        result = mn_model.predict(vector_input)
            
        st.success("Prediction")
        if result == 'Fake':
            st.header("Fake")
        else:
            st.header("Real")

        st.success("Prediction Probabilty")
        result_prob= mn_model.predict_proba(vector_input)
        # st.write(result_prob)
        proba_df = pd.DataFrame(result_prob, columns = mn_model.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["Result","probability"]
        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Result',y='probability',color='Result')
        st.altair_chart(fig,use_container_width=True)

        # add 
# ..................................

    # elif choice == "Bio":
    #     st.subheader("Bio")
    #     def add_bg_from_url():
    #         st.markdown(
    #             f"""
    #      <style>
    #      .stApp {{
    #          background-image: url("https://media.istockphoto.com/id/1278709873/photo/brown-recycled-paper-crumpled-texture-background-cream-old-vintage-page-or-grunge-vignette.jpg?b=1&s=170667a&w=0&k=20&c=NqKmm_gkRwJAqpTbiiqv3TwfWjq9ymOwUDwfG2ck9no=");
    #          background-attachment: fixed;
    #          background-size: cover
    #      }}
    #      </style>
    #      """,
    #      unsafe_allow_html=True
    #      )

    #     add_bg_from_url()
    #     col1, col2, col3  = st.columns([1,1,1])
    #     with col1:
    #         st.image('https://media-exp1.licdn.com/dms/image/C5603AQFHVJRtVAazbw/profile-displayphoto-shrink_800_800/0/1594756980450?e=2147483647&v=beta&t=IiEk7RhiY2zon4WFPFoK4OFTR6Vc31HFPR3lJ076J_8')



    # else:
    #     st.subheader("Resources")
    #     def add_bg_from_url():
    #         st.markdown(
    #             f"""
    #      <style>
    #      .stApp {{
    #          background-image: url("https://media.istockphoto.com/id/1278709873/photo/brown-recycled-paper-crumpled-texture-background-cream-old-vintage-page-or-grunge-vignette.jpg?b=1&s=170667a&w=0&k=20&c=NqKmm_gkRwJAqpTbiiqv3TwfWjq9ymOwUDwfG2ck9no=");
    #          background-attachment: fixed;
    #          background-size: cover
    #      }}
    #      </style>
    #      """,
    #      unsafe_allow_html=True
    #      )

    #     add_bg_from_url()

    # # to_predict = [result]
    # # prediction = model.predict([to_predict])
    # # print(prediction_proba)
    # # value = prediction["Fake"]
    # # if value == "Fake":
    # #     pred_output = 'Fake'
    # #     pred_proba = prediction_proba[0][0].round(2) * 100
    # # else:
    # #     pred_output = 'Real'
    # #     pred_proba = prediction_proba[0][1].round(2) * 100
    # # output_text = '## Predicted a ' + '%' + '**%s chance of %s** \n\n based on the input of %s' % (pred_proba, pred_output, str(to_predict))

if __name__ == '__main__':
    main()


