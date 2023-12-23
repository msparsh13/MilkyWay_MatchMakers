import streamlit as st
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Metric, Rake
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz
import collections
import re
from sklearn.feature_extraction.text import CountVectorizer


from PyPDF2 import PdfReader 
  






stop=stopwords.words('english')


st.title("MilkyWay Matchmakers")

st.header("Upload Resume :")
st.subheader("option 1 : ")
text_1=None

upload_file = st.file_uploader(label="Upload Your Resume in Pdf format", type="pdf")
if upload_file is not None:
    reader = PdfReader(upload_file) 
    text_1=""
    for i in range(len(reader.pages)):

        page = reader.pages[i]
        text_1 += page.extract_text() 

st.subheader("Option 2 : ")

resume_text = st.text_area(label="Write Your Resume")


st.header("Enter Job Description:")
st.subheader("opiton 1 : ")
url = st.text_input('Enter Linkedin URL link')



st.subheader("option 2 :")
job_text = st.text_area(label="Your Job Descripiton")



##now use selenium to go to website and scrap data
##i have to use different scrap bc unsigned noe shows difference 
##i m using linkedin only becasue it  dont require login credentials


## if clicked a button
##headless web srcap
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(options)

text_2=None

if url is not "":
    driver.get(url)
    time.sleep(2)
    driver.find_element(By.CLASS_NAME ,"show-more-less-html__button").click()
    text_2 = driver.find_element(By.CLASS_NAME , "description__text--rich").text 

job_text = job_text or text_2
resume_text = resume_text or text_1

m= st.button(label="click here to get match")


if m and job_text and resume_text :

    try:  ##for geting job
        


        ##Preprocessing and compatiblity


        ##functions to preprocess texts 


        #Removing emojis 

        def remove_emojis(data):
            emoj = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                              "]+", re.UNICODE)
            return re.sub(emoj, '', data)


        ##this function is made to do initial processing of text

        def filter_text(text):
            filter_text = text.lower()
            filter_text= remove_emojis(filter_text)
            filter_words = re.sub(r'[+-=:*|%>,.()>]', ' ' ,filter_text)  ##to remove different signs
            filter_text = filter_text.replace('[^\w\s]','')
            return filter_words

        filtered_job_text = filter_text(job_text)
        filtered_resume_text = filter_text(resume_text)

        ## nlp preprocessing


        # Removing unneeded verbs from text 

        def nlp_preprocessing(text):
            ##tokenization
            words = word_tokenize(text)
            ##pos tagging
            tagged_words = pos_tag(words)
            keywords = [word for word, pos in tagged_words if pos in ['NN', 'NNS'  , 'NNP', 'JJ', 'JJR', 'JJS']]
            return keywords




        def tfidf(text):
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(text)
            feature_names = vectorizer.get_feature_names_out()
            keywords = [feature_names[i] for i in tfidf_matrix.sum(axis=0).argsort()[0 , ::-1]  ][0]
            return keywords



        filtered_job_text = ' '.join(tfidf(nlp_preprocessing(filter_text(job_text)))[0])
        filtered_resume_text = ' '.join(tfidf(nlp_preprocessing(filter_text(resume_text)))[0])

        ##Vectorization

        CV = CountVectorizer()
        CV.fit([filtered_job_text , filtered_resume_text])
        matrix_job = CV.transform([filtered_job_text])
        matrix_resume = CV.transform([filtered_resume_text])


        #Cosine Similarity
        semantic_similarity = cosine_similarity(matrix_job , matrix_resume)

        ##giving matched phrases and find keywords

        tools = ["Python"," R ","SQL","Jupyter","NumPy","Pandas","Matplotlib","Seaborn",
                              "SciPy","Scikit-Learn","TensorFlow","PyTorch","Keras","XGBoost","LightGBM",
                              "Plotly","Dask","Spark","Hadoop","AWS","Google Cloud",
                              "Azure","IBM Watson","NLTK","OpenCV","Gensim","StatsModels",
                              "Theano","Caffe","Keras-Tuner","Auto-Keras","Auto-Sklearn","Shap","ELI5","Bokeh",
                              "Folium","ggplot","plotnine","Geopandas","Datashader","Yellowbrick","H2O.ai","Flask",
                              "Dash","Streamlit","FastAPI","PySpark","TensorBoard","cuDF","NetworkX","BeautifulSoup",
                              "Scrapy","Numba","Cython", "Apache", "Git"]

        Skills = ["Python programming", "Statistics" ,"Probability", "Machine learning","Data visualization","preprocessing" "cleaning","database management", "analysis", "modeling",    
                 "Deep learning","Data engineering", "visualization", "manipulation", "Machine learning", "storage", "Cloud computing", "ETL",    
                 "warehousing","governance", "security","storytelling", "product development", "Natural language processing", "NLP",    
                 "Computer vision", "Business intelligence", "mining","feature engineering", "Time series analysis", "Regression analysis", "Classification algorithms",    
                 "Clustering algorithms", "Neural networks", "Decision trees", "random forests", "Support vector machines", "SVM", "K-nearest neighbors", "KNN", "Reinforcement learning","Hyperparameter tuning",    
                 "Ensemble learning", "Transfer learning", "Unsupervised learning","Supervised learning","Exploratory data analysis", "EDA", "quality control",    
                 "Data interpretation", "Collaboration", "communication", "Project management", "Agile development", "Software engineering", "Version control", "Debugging", "troubleshooting",
                 "Continuous integration and deployment (CI/CD)", "optimization", 'deployment']


        def match_phrases(description, phrases):
            matched_phrase = [phrase for phrase in phrases if fuzz.partial_token_set_ratio(description, phrase) >= 95]
            # Only return matches once
            unique_matches = list(set(matched_phrase))
            return unique_matches


        def extract_education_level(description):
            # Dictionary that maps education levels to their abbreviations
            education_levels = {
                'bachelor': ['bs', 'bachelor'],
                'master': ['ms', 'master'],
                'phd': ['phd'],
                'doctorate': ['doctorate']
            }
            # initialize the education level and maximum ratio to 0
            education_level = None
            max_ratio = 0
            # iterate over the education levels and their abbreviations
            for level, abbreviations in education_levels.items():
                level_variants = [level] + abbreviations
                for variant in level_variants:
                    # calculate the fuzzy matching ratio between the variant and the job description
                    ratio = fuzz.partial_token_set_ratio(variant, description)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        education_level = level
            if max_ratio >= 80:
                return education_level
            else:
                return 'Not Specified'



        def extract_years_of_experience(description):
            # Regular expression pattern to match the years of experience information
            pattern = re.compile(r'(\d+)\s*years?', re.IGNORECASE)

            # search for the pattern in the job description
            match = re.search(pattern, description)

            # if there is a match, return the matched string
            if match:
                return match.group(0)
            else:
                return "Not Specified"


        job_tools = match_phrases(job_text , tools)
        resume_tools= match_phrases(resume_text , tools)

        job_skills= match_phrases(job_text , Skills)
        resume_skills= match_phrases(resume_text, Skills)

        job_qualification= extract_education_level(job_text)
        resume_qualification = extract_education_level(resume_text)

        job_experience = extract_years_of_experience(job_text)
        resume_experience = extract_years_of_experience(resume_text)


        def keyword_match(text1 , text2):
            count_total = len(text2)
            match_keywords  = set(text1) & set(text2)
            count_match = len(match_keywords)
            return count_total, count_match , match_keywords

        ##Compatibility score

        def compatibility_score(cs , tool_match_ratio , skill_match_ratio):
            return 0.2*2*cs+0.4*tool_match_ratio + 0.4*skill_match_ratio


        ## streamlit something to give visual realisations



        ##give list of tools matched
        count_job_tools , count_matched_tools , matched_tools= keyword_match(resume_tools , job_tools)
        count_job_skills , count_matched_skills , matched_skills = keyword_match(resume_skills , job_skills)




        ##show matched tools
        ratio_tools = count_matched_tools/count_job_tools
        ratio_skills = count_matched_skills/count_job_skills



        #Compatiblity score
        Score=compatibility_score(cs=semantic_similarity  , tool_match_ratio=ratio_tools , skill_match_ratio=ratio_skills)



        ## show matched skills

        st.subheader("Your Resume score is (in %):" ) 
        st.write(Score[0][0]*100)

        if Score[0][0] < 0.20:
            st.write("it is :red[BAD] match")

        elif Score[0][0] >=0.20 and Score[0][0] <=0.40:
            st.write("it is :red[AVERAGE] Matching")

        elif Score[0][0]>0.40 and Score[0][0] <=0.65:
            st.write("It is a :green[GOOD] Match") 

        else:
            st.write("It is an :blue[EXCELLENT] Match")

        ##show number of skills matched then show what skills matched  and which one left
        st.header("Skill Matching")

        st.write(count_matched_skills, "skills matched out of" , count_job_skills , "in job descripiton")

        st.subheader("Your Matched Skills")

        st.write(matched_skills)


        st.subheader("Job Skills")
        st.write(job_skills)

        ##show number of tools matched then show what tools matched  and which one left
        st.header("Tool Matching")
        st.write(count_matched_tools , "tools matched out of" , count_job_tools)

        st.subheader("Your Matched tools")
        st.write(matched_tools)

        st.subheader("all tools")
        st.write(job_tools)



        ##Show Qualifacations and experience

        if resume_qualification == job_qualification and resume_qualification is not 'Not Specified':
            st.subheader(":blue[job Qualification Requirement is fulfilled]")
        elif job_qualification is 'Not Specified':
            st.subheader("No qualification requirement")
        else:
            st.subheader("Job Qualification Requirement isnt fulfilled")
            st.write('Job Qualification Requirement' , job_qualification)
            st.write('Your Qualificaiton' , resume_qualification)

        if resume_experience >= job_experience and resume_experience is not 'Not Specified':
            st.subheader(':blue[Job Experience requirement is fulfiled]')
        elif resume_experience is 'Not Specified':
            st.subheader("No experience requirement")
        else:
            st.subheader(":red[Job Experience requirement isnt fulfiled]")
            st.write('Job experience Requirement' , job_experience)
            st.write('Your experience' , resume_experience)

    except:
        st.write("Theres an error")





elif m :
    st.header(":red[Give Job Descriotion link or Resume]")

