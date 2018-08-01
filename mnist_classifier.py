import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import models

dataset_train = pd.read_csv("datasets/MNIST/mnist_train.csv", names=range(785))
dataset_test = pd.read_csv("datasets/MNIST/mnist_test.csv", names=range(785))

X_train = dataset_train.iloc[:,1:]
X_train = X_train.T
X_train /= 255
Y_train = dataset_train.iloc[:,0]

X_test = dataset_test.iloc[:,1:]
X_test = X_test.T
X_test /= 255
Y_test = dataset_test.iloc[:,0]

Y_train_enc = []
for i in range(Y_train.shape[0]):
    temp = np.zeros((10))
    temp[Y_train[i]] = 1
    Y_train_enc.append(temp)
    
Y_test_enc = []
for i in range(Y_test.shape[0]):
    temp = np.zeros((10))
    temp[Y_test[i]] = 1
    Y_test_enc.append(temp)
    
dataset_train = None
dataset_test = None
Y_train = None
Y_test = None

Y_train_enc = pd.DataFrame(Y_train_enc).T
Y_test_enc = pd.DataFrame(Y_test_enc).T

hidden_units = [512,512]
classifier = models.dense_model(X_train.values, Y_train_enc.values, hidden_units, act_fn_list=['relu','relu','softmax'], cost="softmax_cross_entropy_w_logits")


'''
classifier.train(X_train.values, Y_train_enc.values,
                    X_test.values, Y_test_enc.values,
                    learning_rate=0.003,     
                    batch_size=100,  
                    epochs = 32)


classifier.save_weights("results/mnist-weights")
'''

################################### TESTING ON IMAGES #########################
classifier.load_weights("results/mnist-weights")

ix = np.random.randint(0,10000)
X_sample = X_test.iloc[:, ix:ix+1]
Y_sample = Y_test_enc.iloc[:, ix:ix+1]
X_sample_img = X_sample.iloc[:,0].reshape(28,28)
#plt.imshow(X_sample_img)
plt.imsave("test_sample.png", X_sample_img)


X = plt.imread("test_sample.png")
plt.imshow(X)
plt.show()
X = X.mean(axis = 2).reshape(784,1)
X[:,0] -= 0.4

print(np.argmax(classifier.predict(X)))
###############################################################################


y_test_pred = classifier.predict(X_test)
accuracy = np.mean(  np.equal(np.argmax(y_test_pred), np.argmax(Y_test_enc.values)).astype(float)  )
print(accuracy)


