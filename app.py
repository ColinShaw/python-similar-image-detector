from flask           import Flask
from src.image_match import ImageMatch


app = Flask(__name__)
im  = ImageMatch()

@app.route('/match', methods=['POST'])
def match():
    return im.match()

