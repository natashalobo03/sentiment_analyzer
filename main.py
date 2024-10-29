from flask import Flask,request,jsonify,render_template # type: ignore
from nltk.sentiment.vader import SentimentIntensityAnalyzer # type: ignore
import nltk # type: ignore

app=Flask(__name__)

nltk.download('vader_lexicon')
sid=SentimentIntensityAnalyzer()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    data=request.json
    text=data['text']
    sentiment_scores=sid.polarity_scores(text)
    if sentiment_scores['compound']>= 0.05:
       sentiment="Positive"
    elif sentiment_scores['compound']<=-0.05:
       sentiment="Negative"
    else:
       sentiment="Neutral" 
        
    
    return jsonify({'Sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)     