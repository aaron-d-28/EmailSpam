import pickle
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Only needed once
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stoppingwords = stopwords.words('english')


def transformtext(text:str):
    text = text.lower()
    text = nltk.word_tokenize(text)

    strings = [i for i in text if i.isalnum()]
    strings = [i for i in strings if i not in stoppingwords and i not in string.punctuation]
    strings = [ps.stem(i) for i in strings]

    return ' '.join(strings)


app = FastAPI()

# Load vectorizer
with open('vectorizer_pickle.pkl', 'rb') as f:
    tf = pickle.load(f)

# Load model
with open('random_forest_model.pkl', 'rb') as f:
    RandomForest = pickle.load(f)


class TextInput(BaseModel):
    Text: str
@app.get("/")
def read_root():
    strings =list(tf.vocabulary_.keys())[:10]
    return {"message": "Welcome to the backend server!","vocab":strings}


@app.get("/Classify/")
def read_item(Message: TextInput):
    transformedtxt = transformtext(Message.Text)
    vector = tf.transform([transformedtxt])
    result = RandomForest.predict(vector)
    print(result)
    return {"prediction": "Spam" if result[0] == 1 else "Ham"}


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000)
