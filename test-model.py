# Save this as test_model.py in your cropguard folder and run it
import numpy as np
import tensorflow as tf
from keras.preprocessing import image as kimg
from keras.applications.mobilenet_v2 import preprocess_input as mv2_pre

# Class names exactly as your dataset trained them
CLASS_NAMES = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
}

# Load model
model = tf.keras.models.load_model('mobilenet_best.h5')

# Test with any leaf image you have — change this path
IMG_PATH = 'test-leaf.jpg'  # put any leaf image in your cropguard folder

img = kimg.load_img(IMG_PATH, target_size=(224, 224))
arr = kimg.img_to_array(img)
arr = np.expand_dims(arr, axis=0)
arr = mv2_pre(arr)

preds = model.predict(arr, verbose=0)[0]
top5  = np.argsort(preds)[::-1][:5]

print("\n── MobileNetV2 Predictions ──")
for i in top5:
    print(f"  {CLASS_NAMES[i]:<50} {preds[i]*100:.2f}%")