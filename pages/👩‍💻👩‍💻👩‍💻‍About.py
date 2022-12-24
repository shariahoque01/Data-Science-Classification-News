import streamlit as st
st.markdown("""
<style>
.About {
    font-size:50px !important;
    text-align: center;
    color: black
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="About">Know More About Us 👩‍💻👩‍💻👩‍💻 </p>', unsafe_allow_html=True)

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
        st.image('images/tahira1.jpg')
        important_links = '''
        [github](https://github.com/tahiratabassum19),
        [LinkedIn](www.linkedin.com/in/tahira-tabassum),
        '''

        st.markdown("""
        <style>
        .Tah {
            font-size:15px !important;
            text-align: left;
            font-family:Gothic;
            font-weight: bold;
            color: darkblue
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="Tah">I am Tahira Tabassum majoring in Computer Science at Queens College.I will be graduating this semester. I am enthusiastic about data science, software development, Problem Solving, and Front End Design. Fun Fact: I am scared of dogs</p>', unsafe_allow_html=True)
        st.markdown(important_links)
    with col2:
        st.image('images/Jennifer1.jpg')
        important_links1 = '''
        [github](https://github.com/jenalvarado),
        [LinkedIn](https://www.linkedin.com/in/jenyalvarado/),
        '''


        st.markdown("""
        <style>
        .Tah {
            font-size:15px !important;
            text-align: left;
            font-family:Times New Roman;
            font-weight: bold;
            color: darkblue
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="Tah">I am Jennifer Alvarado majoring in Creative Writing with a Minor in Computer Science at Brooklyn College.My interests are Writing,UI/UX Design, Hiking, and Cooking. I want to acquire a frontend position in designing web applications/mobile apps. Fun Fact:My right pinky can bend further back than my left one</p>', unsafe_allow_html=True)
        st.markdown(important_links1)

    with col3:
        st.image('images/Sharia.jpg')
        important_links2= '''
        [github]( https://github.com/shariahoque01),
        [LinkedIn](https://www.linkedin.com/in/sharia-hoque/),
        '''
        st.markdown("""
        <style>
        .Tah {
            font-size:15px !important;
            text-align: left;
            font-family:Gothic;
            font-weight: bold;
            color: darkblue
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="Tah">I am a senior undergradute student and majoring in Statistics and Quantitative Modeling at Baruch College Zicklin School of Business (CUNY). I have a keen interest in Data, technology and Business. Fun fact: I like Weightlifting and exploring Arts.</p>', unsafe_allow_html=True)
        st.markdown(important_links2)
add_bg_from_url() 