from tespitv2 import emotion_recognition
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from flask_cors import CORS
from flask import Flask, make_response, jsonify
import tespitv2
import json

app = Flask(__name__)
CORS(app)

print("DENEME -------- ")




@app.route('/start', methods=['POST'])
def start2():
    tespitv2.recording = False
    res = emotion_recognition()
    print(res)
    json.dumps(res)
    return res

@app.route('/close', methods=['POST'])
def start3():
    tespitv2.recording = True
    return jsonify(tespitv2.recording)

if __name__ == '__main__':
    app.run(debug=True) 
    



    