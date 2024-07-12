import os
import tensorflow
from flask import Flask, request, render_template, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

import sys
sys.stdout.reconfigure(encoding='utf-8')

app=Flask(__name__)
model=tensorflow.saved_model.load("model_3")

print('Model Loaded!')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST','GET'])
def predictor():
    if request.method=='POST':
        f=request.files['image']
        
        basepath=os.path.dirname(__file__)
        
        #retrieving directory path of current python script.
        #we will store image uploaded by user in uploads folder.
        
        print(basepath)
        
        #concatenatin uploads to basepath so we can create uploads folder
        #to store uploaded files
        
        filepath=os.path.join(basepath,'uploads',f.filename)
        print(filepath)
        
        f.save(filepath)
        
        # Read and preprocess the image
        img=load_img(filepath,target_size=(224,224))
           
        x = img_to_array(img)
        
        # Make predictions using the model
        predictions = model(np.array([x]))
        print(predictions)
        
        pred_prob = predictions[0][0].numpy()  # Assuming binary classification
        print(pred_prob)
        
        # Determine class name based on prediction probability
        class_names = ['cataract', 'normal']
        class_index = 1 if pred_prob > 0.5 else 0
        result_text = f"Result is {class_names[class_index]} with confidence {pred_prob:.2f}"
        
        return render_template('result.html', result=result_text)
        
        
        
    return render_template('details.html')
if __name__=='__main__':
    app.run(debug=True, port=8001)