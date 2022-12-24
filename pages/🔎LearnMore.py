import streamlit as st
st.subheader("LearnMore")
st.markdown("""
<style>
.About {
    font-size:50px !important;
    text-align: center;
    color: black
}
</style>
""", unsafe_allow_html=True)

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
   
    col1, col2, col3  = st.columns([1,1,1])
    with col1:
        st.image('images/smoke.jpg',use_column_width='always')
        st.markdown("""
        <style>
        .smoke {
            font-size:15px !important;
            text-align: left;
            font-family:Sans-serif;
            font-weight: bold;
            color: black

        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="smoke">Can We Trust Social Media as a News Source?</p>', unsafe_allow_html=True)
        st.write('<a href="https://blog.miappi.com/reliability-of-social-media-news-source">View Article</a>', unsafe_allow_html=True)
        st.image('images/trump1.jpg',use_column_width='always')
        st.write()
        st.markdown("""
        <style>
        .trump1 {
            font-size:15px !important;
            text-align: left;
            font-family:Sans-serif;
            font-weight: bold;
            color: black

        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="trump1">Pope Francis endorsed President Trump </p>', unsafe_allow_html=True)
        st.write('<a href="https://www.science.org/content/article/majority-americans-were-not-exposed-fake-news-2016-us-election-twitter-study-suggests">View Article</a>', unsafe_allow_html=True)
        st.write()



        
    with col2:

    # display the link to that page.
        # display another picture
        #Bleach
        st.image('images/bleach1.jpg')
        st.markdown("""
        <style>
        .bleach {
            font-size:14px !important;
            text-align: left;
            font-family:Sans-serif;
            font-weight: bold;
            color: black
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="bleach">It’s been exactly one year since Trump suggested injecting bleach.</p>', unsafe_allow_html=True)
        st.write('<a href="https://www.politico.com/news/2021/04/23/trump-bleach-one-year-484399">View Article.</a>', unsafe_allow_html=True)
        

        st.image('images/candy.jpg',use_column_width='always')
        st.markdown("""
        <style>
        .candy {
            font-size:15px !important;
            text-align: left;
            font-family:Sans-serif;
            font-weight: bold;
            color: black

        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="candy">Woman murders roommate for sending too many Candy Crush requests</p>', unsafe_allow_html=True)
        st.write('<a href="https://bdc-tv.com/woman-murders-college-roommate-sending-many-candy-crush-requests/">View Article</a>', unsafe_allow_html=True)

    with col3:
        st.image('images/nasa.jpg',use_column_width='always')
        st.markdown("""
        <style>
        .nasa {
            font-size:15px !important;
            text-align: left;
            font-family:Sans-serif;
            font-weight: bold;
            color: black

        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="nasa">NASA Warns Planet X Is Headed Straight for Earth?</p>', unsafe_allow_html=True)
        

        st.write('<a href="https://www.snopes.com/fact-check/nasa-warns-nibiru-is-headed-straight-for-earth/">View Article</a>', unsafe_allow_html=True)

        














        
        
        #Harambe
        st.image('images/Harambe.jpg',use_column_width='always')
        st.markdown("""
        <style>
        .harambe {
            font-size:15px !important;
            text-align: left;
            font-family:Sans-serif;
            font-weight: bold;
            color: black

        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="harambe">Harambe won thousands of votes in the US election</p>', unsafe_allow_html=True)
        st.write('<a href="https://www.bbc.com/news/blogs-trending-37925961">View Article.</a>', unsafe_allow_html=True)

add_bg_from_url() 
