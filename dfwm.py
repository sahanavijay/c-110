import cv2
import numpy as np
import tensorflow as tf
model= tf.keras.models.load_model('keras_model.h5')

video=cv2.VideoCapture(1)
while True:
    check, frame = video.read()
    img=cv2.resize(frame,(224,224))
    test_img=np.array(img,dtype=np.float32)
    test_img=np.expand_dims(test_img,axis=0)
    normalized_img=test_img/255.0
    prediction=model.predict(normalized_img)
    print("prediction: ",prediction)
    cv2.imshow("result",frame)
    key=cv2.waitKey(1)
    if key==32:
        print("closing")
        break
video.release()