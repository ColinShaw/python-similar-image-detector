from flask           import Flask, request, jsonify
from scipy.misc      import imread
from io              import BytesIO
from src.image_match import ImageMatch


app = Flask(__name__)
im  = ImageMatch()

@app.route('/match', methods=['POST'])
def match():
    image = request.files['image'].read()
    image = imread(BytesIO(image)) 
    label = im.match(image)
    return jsonify(label=label)

