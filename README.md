# Milkyway-Matchmakers
## Overview and Purpose
 AI-powered job matching system that analyzes job descriptions and candidate profiles to recommend the best matches.The system makes it easier for candidate to match job descripiton with his resume to know matching score.
### Key Features:
1. **Natural Language Processing (NLP)**: The system utilizes advanced NLP algorithms to comprehend and interpret the nuances of both job descriptions and candidate profiles. This allows for a more sophisticated understanding of the requirements and qualifications, going beyond keyword matching.
2. **Contextual Understanding**: By employing contextual analysis, the system gains a deeper understanding of the context in which certain skills, experiences, and qualifications are mentioned. This results in more accurate and nuanced job-candidate matches.
3. **Semantic Matching**: The AI system goes beyond traditional keyword matching by employing semantic analysis. It identifies relationships and similarities between different terms, enabling the recommendation of candidates who possess not only the exact skills but also related competencies that may be valuable to the employer.

## Dependencies:
1. Pandas (2.0.3)
2. seaborn (0.12.2)
3. matplotlib (3.7.1)
4. rapidfuzz (3.5.2)
5. nltk  (3.7)
6. sklearn (1.2.2)
7. torch (2.1.1)
8. sentence_transformers (2.2.2)
9. selenium  (4.16.0)
10. streamlit (1.25.0)
11. PyPDF2 (3.0.1)
12. rake_nltk (1.0.6)

## Installation:
```
pip install pandas==2.0.3
pip install seaborn==0.12.2
pip install matplotlib==3.7.1
pip install rapidfuzz==3.5.2
pip install nltk==3.7
pip install scikit-learn==0.24.2 
pip install torch==2.1.1  
pip install sentence-transformers==2.2.2
pip install selenium==4.16.0
pip install streamlit==1.25.0
pip install PyPDF2==3.0.1
pip install rake_nltk==1.0.6
```
## Usage Guideline:
- **Step 1 :** Run Streamlit Application using  streamlit run ui.py
- **Step 2 :** Either Provide Resume Text or Resume in pdf format
- **Step 3 :** Similarly Provide Job Description text or __Linkden__ url
- **Step 4 :** Click on Button 

## Organisation
Organisation of code is as given follows:
```
/project-root
||-- job_offers.csv
||-- EDA.ipynb
||-- ui.py
||==README.md
```

- **job_offers.csv** : It is dataset created by Web Scrapping __linkedin__ Jobs page
- **EDA.ipynb** : It is explanotary data analysis of job_offers dataset 
- **ui.py** : It is streamlit application

## Contact:
-  __Email__ : msparsh07@gmail.com

#
