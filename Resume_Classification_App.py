#Importing Libraries
import pandas as pd
#import numpy as np
import warnings
warnings.filterwarnings("ignore")

import re  #regular expression

import nltk
nltk.download('stopwords')
from textblob import Word
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
import streamlit as st
#import sklearn
#from xgboost import XGBClassifier
import pickle
import docx2txt
from PyPDF2 import PdfFileReader
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.ensemble import RandomForestClassifier

# load file and create model

classification_model = pickle.load(open(r'XGBmodel.pkl', 'rb'))
features = pickle.load(open(r'feature.pkl', 'rb'))
print('Loaded model from file')

# function to preprocess text
def preprocess(x):

    # removing both the leading and the trailing characters such as spaces in tweets
    x = x.strip()

    #Removing spaces in between
    x=re.sub(r'\s+', " ", x)
    #To remove special characters and numbers
    #allow a-z A-Z and space character only, other than that replace it with null
    x=re.sub('[^a-zA-Z ]', '', x)

    #converting into lower case characters
    x=x.lower()

    #spiliting
    x=x.split()

    #removing stop words
    x=[word  for word in x if word not in set(stopwords.words('english'))]

    #Lemmatization
    x=[Word(word).lemmatize() for word in x]
        
    #joining
    x=" ".join(x)
    return x

def main():


    # giving a title
    st.title('Resume Classification Application')
    resume_text=[]
    file_name=[]
    uploaded_files = st.file_uploader("Choose a .docx file", type=['docx','pdf'], accept_multiple_files=True )
     
    
    submit = st.button('Predict Resume class/category')
    if submit:

        if uploaded_files is  None:
            
            st.session_state["upload_state"] = "Upload a file first!"
        else :
            for uploaded_file in uploaded_files:
                #File Name
                if uploaded_file.type == "application/pdf" :
                    pdfReader = PdfFileReader(uploaded_file)
                    count = pdfReader.numPages
                    text = ""
                    for i in range(count):
                         page = pdfReader.getPage(i)
                         text += page.extractText()
	
                else :     
                                      
                    #Reading the document text        
                    text=docx2txt.process(uploaded_file)
                file_name.append(uploaded_file.name)
                resume_text.append(text)
            
            #Creating DataFrame for uploaded resume
            resume_data=pd.DataFrame()
            resume_data['File_Name']=file_name
            resume_data['Resume_Text']=resume_text
            
            #Preprocess resume Text
            resume_data['Resume_Text']=resume_data["Resume_Text"].apply(preprocess)
            
            #Vectorizer to extract features
            vectorizer = TfidfVectorizer(vocabulary=features)
            X = vectorizer.fit_transform(resume_data.Resume_Text.values).toarray()
            
            Y = classification_model.predict(X)
            resume_data['Category']=Y
            
            resume_data['Category'] = resume_data['Category'].map({0: 'People soft', 1 : 'React JS Developer', 2:'SQL Developer', 3:' Work Day'})

    
            st.subheader("Resume Category:")
            for index, row in resume_data.iterrows():  
                st.write(row['File_Name'] ,' : ', row['Category'] )
            st.write('This application classifies Resumes only in Categories : PeopleSoft, React JS Developer, SQL Developer and Workday' ) 
        
        
        
if __name__ == '__main__':
    main()

