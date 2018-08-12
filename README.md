# cortopy
Deep Neural Network framework module built from scratch with numpy 

---

## Currently available options:

**Classes:**   

                    1. dense_model


**Activation functions:**

                    1. ReLU
                    2. Sigmoid                         
                    3. Hyperbolic tangent                       
                    4. Softmax
                       
**Losses:**              

                    1. Mean squared error
                    2. Binary cross entropy
                    3. Softmax cross entropy (multiclass)
                       
**Optimizers:**      

                    1. Mini-batch Gradient Descent
                    2. Gradient Descent with momentum
                    3. RMS prop
                    4. ADAM
                      
---  

## Usage in code:

#### Import "models" and create a dense_model object:
```python
import models
model = models.dense_model(X_train, Y_train_enc, hidden_units, act_fn_list, cost)
```
Options:  
*act_fn_list* - 'relu', 'sigmoid', 'tanh', 'softmax'  
*cost* - 'mse', 'binary_cross_entropy', 'softmax_cross_entropy_w_logits'

#### Train your model:
```python
classifier.train( X_train, Y_train_enc,
                  X_test, Y_test_enc,
                  learning_rate,     
                  batch_size,  
                  epochs,
                  optimizer,
                  momentum_beta=0.9,
                  rmsprop_beta=0.999 )
```
The training process returns the Loss v/s Epoch plot


Options:
*optimizers* - 'minibatch_GD', 'momentum_GD', 'RMS_prop', 'ADAM'


#### Make predictions:
```python
prediction = model.predict(X_sample)
```

#### Save weights:
```python
model.save_weights(path)
```

#### Load weights into the model:
```python
model = models.dense_model(X_train-like, Y_train_enc-like, hidden_units, act_fn_list, cost)
model.load_weights(path)
```
---

## MNIST handwritten digits dataset
This module was used to create a classifier for the MNIST dataset with the following sets of configurations:

#### Architecture:
      No. of hidden  layers: 2
      No. of hidden units: [512,512]
      Activation functions: ['relu','relu','softmax']
      
##### Configuration-1:      
      Learning rate: 0.003
      Batch size: 100
      No. of epochs: 32
      Optimizer: Mini-batch Gradient Descent

Result: 

![alt text](https://github.com/c0rtis-prime/cortopy/blob/master/results/mnist-1/mnist-Loss_plot_%5Boptmzr%3DminibatchGD%5D_%5Blr%3D0.003%5D.png.png "Loss plot") 
      
      Train accuracy = 92.691667%
      Test accuracy = 92.65% 

##### Configuration-2:     
      Batch size: 100
      No. of epochs: 32
      Optimizer: Gradient Descent with momentum
      Learning rate: 0.03
      momentum_beta = 0.9

Result: 

![alt text](https://github.com/c0rtis-prime/cortopy/blob/master/results/mnist-2/mnist-Loss_plot_%5Boptmzr%3DmomentumGD%5D_%5Blr%3D0.03%5D.png "Loss plot") 
      
      Train accuracy = 95.71%
      Test accuracy = 95.23%

##### Configuration-3:      
      Batch size: 100
      No. of epochs: 25
      Optimizer: RMS prop
      Learning rate: 0.0003
      Decay = 0.99

Result: 

![alt text](https://github.com/c0rtis-prime/cortopy/blob/master/results/mnist-3/Loss_plot_%5Boptmzr%3DRMS_prop%5D_%5Blr%3D0.0003%5D.png "Loss plot") 
      
      Train accuracy = 99.921667%
      Test accuracy = 97.88%
      
 #### Configuration-4:    
      Batch size: 100
      No. of epochs: 52
      Optimizer: ADAM
      Learning rate: 0.0003
      RMS decay = 0.999
      momentum_beta = 0.9

Result: 

![alt text](https://github.com/c0rtis-prime/cortopy/blob/master/results/mnist-4/Loss_plot_%5Boptmzr%3DADAM%5D_%5Blr%3D0.0001%5D.png "Loss plot") 
      
      Train accuracy = 99.941667%
      Test accuracy = 97.9%
