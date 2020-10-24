from flask import Flask, request, jsonify
import car_damage
from cv2 import cv2
import time
import numpy as np
import base64
app = Flask(__name__)


@app.after_request
def add_header(response):

# response.cache_control.no_store = True
    response.headers['Cache-Control'] ='no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] ='-1'

    return response


# @app.route('/')
# def home():
#     context = {}
#     context['img'] = 'static/imgages/download.png'
#     #context['num'] = 0
#     return render_template('index.html',context = context)


# @app.route('/upload', methods = ['POST'])
# def upload():
#     context = {}
#     img = request.files.get('file')
#     img.save('static/imgages/testfile.jpg')
#     context['img'] = 'static/imgages/testfile.jpg'
#     #context['num'] = 0 
#     return render_template('index.html',context = context)

# @app.route('/detect_damage', methods = ['GET'])
# def detect_damage():
#     context = {} 
#     image_path = 'static/imgages/testfile.jpg'
#     image = cv2.imread(image_path)
#     predictions = car_damage.getDefectsInfo(image)
#     for prediction in predictions:
#         image = cv2.rectangle(image, (int(prediction['bbox'][0]), int(prediction['bbox'][1])), (int(prediction['bbox'][2]), int(prediction['bbox'][3])), (np.random.choice(256), np.random.choice(256), np.random.choice(256)), 4)
#     cv2.imwrite('static/imgages/annotated_image.jpg', image)
#     context['img'] = 'static/imgages/annotated_image.jpg'
#     context['data'] = predictions
#     context['num'] = len(context['data'])
#     context['show_damage'] = True
#     #print(context)
#     return render_template('index.html', context = context)

@app.route('/api/v1/getdamage', methods = ['POST'])
def get_damage():

    # Receiving and saving the file in which the defect is to be find.
    img = request.files.get('file')
    timestamp = str(time.time()).split('.')[0]
    image_path = 'uploads/unprocessed/image_' + timestamp + '.jpg'
    img.save(image_path)

    # Finding the damage 
    image = cv2.imread(image_path)
    predictions = car_damage.getDefectsInfo(image)
    for prediction in predictions:
        image = cv2.rectangle(image, (int(prediction['bbox'][0]), int(prediction['bbox'][1])), (int(prediction['bbox'][2]), int(prediction['bbox'][3])), (np.random.choice(256), np.random.choice(256), np.random.choice(256)), 4)
        del prediction['bbox']

    # Creating response
    _, imencoded = cv2.imencode(".jpg", image)
    im_b64 = base64.b64encode(imencoded)
    response = {}
    response['image'] = im_b64.decode()
    response['damages_info'] = predictions

    return jsonify(response)


if __name__ == '__main__':
    app.run()
