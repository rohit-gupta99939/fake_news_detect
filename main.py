# 1. Library imports
import uvicorn
from fastapi import FastAPI,Request
from FakeNewsClass import NewsClass
import text_to_row
import numpy as np
import pickle
import pandas as pd
import random 
import string
import lime
from lime import lime_text
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)
pickle2_in = open("simple_classifier.pkl","rb")
simple_classifier=pickle.load(pickle2_in)

templates = Jinja2Templates(directory="templates")
origins = [
    "*"
]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:NewsClass):
    data = data.dict()
    tit=data['tit']
    text=data['text']
    row=text_to_row.row_gen(tit,text)
    col=['text_length','title_length','text_char_length','unique_word','avg_wordlength','unique_vs_words','number_of_link','number_of_noun','number_of_verb','number_of_adj','All_text']
    test = pd.DataFrame([row],columns=col)
    res=classifier.predict(test)
    prob=classifier.predict_proba(test)
    class_names = ["True", "Fake"]
    file_name = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    file_name =file_name+'.html'
    explainer =lime_text.LimeTextExplainer(class_names = class_names)
    exp = explainer.explain_instance(test['All_text'][0], simple_classifier.predict_proba, num_features = 15)
    exp.save_to_file('graph/'+file_name)
    #print(prob[0][0])
    if(res[0]==0):
        prediction='True news'
        p_prob=round(prob[0][0]*100,2)
    else:
        prediction='Fake news'
        p_prob=round(prob[0][1]*100,2)
    return {
        'prediction': prediction,
        'prob':p_prob,
        'graph': file_name
    }
   # print(l)
    '''data={
        'tit':tit,
        'text':text
    }'''
    '''return {
        'data': l
    }'''
    '''
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }'''

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
 #   uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload