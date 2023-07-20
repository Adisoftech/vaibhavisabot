from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

app = Flask(__name__)

# Load NLTK data and other initializations
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

conversation_history = []  # To store the conversation history
past_questions = []  # To store past asked questions


def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res


@app.route("/")
def index():
    return render_template("index.html", conversation_history=conversation_history, past_questions=past_questions)


@app.route("/run_command", methods=["POST"])
def run_command():
    data = request.json
    if "command" in data:
        command = data["command"]
        response = chatbot_response(command)
        conversation_history.append({"user": command, "bot": response})
        return jsonify({"response": response})

    if "voice" in data:
        voice_input = data["voice"]
        response = chatbot_response(voice_input)
        conversation_history.append({"user": voice_input, "bot": response})
        past_questions.append(voice_input)
        return jsonify({"response": response})

    return jsonify({"response": "Error: Invalid Input"})


if __name__ == "__main__":
    app.run(debug=True)
