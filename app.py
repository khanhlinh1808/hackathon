from flask import Flask, render_template, request
import os
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder

dic = {0: 'acne-and-rosacea', 1: 'atopic-dermatitis', 2: 'eczema', 3: 'herpes-zoster', 4: 'lichen-planus', 5: 'nail-fungus', 6: 'other',7: 'psoriasis',8: 'tinea',9: 'urticaria'}
dic1 = {0: 'static/data/0Mụn trứng cá/dinh nghia.PNG', 1: 'static/data/1Viêm da cơ địa/dinhnghia.PNG',
        2: 'static/data/2Chàm/dinhnghia.PNG', 3: 'static/data/3Zona/dinhnghia.PNG',
        4: 'static/data/4Lichen phẳng/dinhnghia.PNG', 5:'static/data/5Nấm móng/dinhnghia.PNG',
        6: 'static/data/6Bệnh khác/dinhnghia.PNG', 7: 'static/data/7Vảy nến/dinhnghia.PNG',
        8: 'static/data/8Nấm da đầu/dinhnghia.PNG', 9: 'static/data/9Mề đay/dinhnghia.PNG' }

dic2 = {0: 'static/data/0Mụn trứng cá/tuvan.PNG', 1: 'static/data/1Viêm da cơ địa/tuvan.PNG',
        2: 'static/data/2Chàm/tuvan.PNG', 3: 'static/data/3Zona/tuvan.PNG',
        4: 'static/data/4Lichen phẳng/tuvan.PNG', 5:'static/data/5Nấm móng/tuvan.PNG',
        6: 'static/data/6Bệnh khác/tuvan.PNG', 7: 'static/data/7Vảy nến/tuvan.PNG',
        8: 'static/data/8Nấm da đầu/tuvan.PNG', 9: 'static/data/9Mề đay/tuvan.PNG' }

model = load_model('resnet50.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def home():
  return render_template('AIMed1.html')

@app.route('/', methods=['POST'])
def predict():
    # predicting images
    imagefile = request.files['imagefile']
    image_path = 'static/images/' + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)

    classes = model.predict(x)
    result = np.argmax((classes[0]))
    pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    print(classes[0])
    print(tf.nn.sigmoid(classes[0]))
    print(result)
    print(dic[result])

    contents = Image.open('{}'.format(dic2[result]))

    return render_template('AIMed1.html', user_image=pic,
                           prediction_text='{}'.format(dic[result]),
                           definition_image='{}'.format(dic1[result]),
                           advice_image='{}'.format(dic2[result]))

if __name__ == '__main__':
    app.run()
