from flask import Flask, request, jsonify
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload and process directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER



# Load your pre-trained model
model = load_model('finger_vgg_84.h5')
model_loop=load_model('loop.h5')


# Create the 'uploads' and 'processed' directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/image-selector', methods=['POST'])
def image_selector():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image to the 'uploads' directory
        filename = secure_filename(file.filename)
        save_dir=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the image using your model
        # Replace this with your actual image processing code
        result = process_image(save_dir)

        # Move the image to the 'processed' directory
        try:
            os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['PROCESSED_FOLDER'], filename))
        except FileExistsError:
            pass            

        return jsonify({'result': result})

def process_image(filename):
    # Replace this with your image processing logic using the loaded model
    #Loading the image
    img = keras.preprocessing.image.load_img(filename, target_size=(640, 720))
    #converting it into array 
    x2=keras.preprocessing.image.img_to_array(img)
    x=keras.preprocessing.image.img_to_array(img)
    ## Scaling
    x=x/255
    x2=x2/255
    x2=np.expand_dims(x2, axis=0)
    x=np.expand_dims(x, axis=0)
    # calling the first model and passing the image for prediction
    preds1= model.predict(x)
    preds1=np.argmax(preds1, axis=1)
    class_name2=''
    if preds1[0]==0:
        class_name='arch'
    elif preds1[0]==1:
        class_name='loop'
        #calling the second model to check for ulnar or radial loop
        pred2=model.predict(x2)
        pred2=np.argmax(pred2,axis=1)
        if pred2[0]==0:
            class_name='radial loop'
        elif pred2[0]==1:
            class_name='ulnar loop'        
    elif preds1[0]==2:
        class_name='whorl'
    else:
        class_name='unknown'

    return f'Processed {class_name}'

if __name__== '__main__':
    app.run(debug=True,host="0.0.0.0",port=8000)
