import cv2 as cv
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from PIL import Image
import os

app = Flask(__name__, template_folder='template', static_folder='static')

#definig upload folder path
UPLOAD_FOLDER = './web/static/predicted_image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Secret key to enable session
app.secret_key = 'This is a secret key to utilize session in Flask'

model = tf.keras.models.load_model('./object_localization.h5')

#testing
@app.route('/index')
def index():
    return 'hello world'

@app.route('/')
def home():
    return render_template('interface.html')

#get image from frontend

@app.route('/objectLocalization', methods=['POST'])
def objectLocalization():
    #get and save image
    imagefile = request.files['imagefile']
    path = os.path.join('./web/static/image_temp', imagefile.filename)
    print(path)
    imagefile.save(path)

    #read image and predict
    image = cv.imread(path)
    # print(type(image))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))

    label, bounding_box = model.predict(np.expand_dims(image, axis = 0))

    confidence_value = label[0][np.argmax(label)]

    label_pred = np.argmax(label)

    if label_pred == 0:
        label_name = 'cucumber'
    elif label_pred == 1:
        label_name = 'eggplant'
    else:
        label_name = 'mushroom'

    text = '{}: {percent:.0%}'.format(label_name, percent=confidence_value)

    #put text and create bounding box
    cv.rectangle(image, (int(bounding_box[0][0]), int(bounding_box[0][1])), (int(bounding_box[0][2]), int(bounding_box[0][3])), (255, 0, 255), 3)
    cv.putText(image, text,  (int(bounding_box[0][0]), int(bounding_box[0][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1, 255)
    # cv.imwrite('./web/predicted_image/predicted_image.jpg', image)
    
    #save image to app.config
    image_pil = Image.fromarray(image)
    image_pil.save(os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_image.jpg'))
    
    #store uploaded image in flask session
    # session['uploaded_predicted_image'] = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_image.jpg')

    show_image = os.path.join(app.config['UPLOAD_FOLDER'],'predicted_image.jpg')
    
    json_response = {'prediction': label_name,
                    'confidence_value': '{percent:.0%}'.format(percent=confidence_value),
                    }
    
    return render_template('interface.html')

    #use return render_template(interface.html, predicted_image = show_image)  is an another option if static folder not inside another folder

# @app.route('/show_predicted_image', methods = ['GET'])
# def show_predicted_image():
#     image_predicted_path = session.get('uploaded_predicted_image', None)

#     return render_template('show.html', predicted_image = image_predicted_path)
    

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=2000)