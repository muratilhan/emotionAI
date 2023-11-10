from tespit import deneme
import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from flask_cors import CORS
from flask import Flask, make_response, jsonify

app = Flask(__name__)
CORS(app)



@app.route('/start', methods=['POST'])
def start():
    result = deneme()
    response = jsonify(result)
    return response


    
if __name__ == '__main__':
    app.run(debug=True)
    



    