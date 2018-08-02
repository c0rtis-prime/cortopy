import numpy as np
import cv2
import cortopy.models as models


X_train = np.zeros([784,60000])
Y_train_enc = np.zeros([10,60000])

hidden_units = [512,512]
classifier = models.dense_model(X_train, Y_train_enc, 
                                hidden_units, 
                                act_fn_list=['relu','relu','softmax'], 
                                cost="softmax_cross_entropy_w_logits")

classifier.load_weights("results/mnist-weights")

X_train = None
Y_train_enc = None

def predict(img_array):
    return np.argmax(classifier.predict(img_array))


cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    #x, y, w, h = 0, 0, 300, 300
    x, y, w, h = 424-150, 240-150, 300, 300
    
    ret, img = cam.read()
    
    prediction = ''
    
    if ret:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        scratch_area = img[y:y+h, x:x+h]
        scratch_area = cv2.cvtColor(scratch_area, cv2.COLOR_BGR2GRAY)
        
        _, scratch_area = cv2.threshold(scratch_area,15,255, cv2.THRESH_BINARY_INV)
        
        input_img = cv2.resize(scratch_area, (28,28))
        input_img = np.array(input_img).reshape(784,1)
        
        prediction = predict(input_img)
        
        cv2.putText(img, 
                    "Prediction: "+ str(prediction), 
                    (x,450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)
        
        cv2.imshow("Frame", img)
        cv2.imshow("scratch area", scratch_area)
        
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
