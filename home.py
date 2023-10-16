#Importing libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st

def loop_model(img):
    model1=load_model('loop.h5')
    img = image.load_img(img, target_size=(640, 720))

    # Preprocessing the image
    x=image.img_to_array(img)
    ## Scaling
    x=x/255
    x=np.expand_dims(x, axis=0)
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)
    preds = model1.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds[0]==0:
        class_name2='radial loop'
    elif preds[0]==1:
        class_name2='ulnar loop'
    return class_name2




def load_and_display(img):
    model=load_model('finger_vgg_84.h5')
    new_img=img
    img = image.load_img(img, target_size=(640, 720))

    # Preprocessing the image
    x=image.img_to_array(img)
    ## Scaling
    x=x/255
    x=np.expand_dims(x, axis=0)
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    class_name2=''
    if preds[0]==0:
        class_name='arch'
    elif preds[0]==1:
        class_name='loop'
        class_name2=loop_model(new_img)
    elif preds[0]==2:
        class_name='whorl'
    else:
        class_name='unknown'
    
    return class_name,class_name2


#Front end
st.header('Finger Print Classifier')

col1,col2=st.columns(2)
upload_stat=0
with col1:
    
    input_image=st.file_uploader('Upload The Image File Here',type=['jpg','png'])
    
    confirmation=st.button('Submit')
    if confirmation==True:
        st.image(input_image)
        upload_stat=1

with col2:
    st.subheader('Detection:')
    if upload_stat==1:
        img_uploaded_class,detail_class=load_and_display(input_image)
        st.subheader(img_uploaded_class)
        if len(detail_class)>1:
            st.subheader(detail_class)

