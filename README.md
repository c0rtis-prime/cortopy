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
classifier.train(X_train, Y_train_enc,
                 X_test, Y_test_enc,
                 learning_rate,     
                 batch_size,  
                 epochs
                 optimizer)
```
The training process returns the Loss v/s Epoch plot


Options:
*optimizers* - 'minibatch_GD', 'momentum_GD'


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

#### Configuration-1:
      No. of hidden  layers: 2
      No. of hidden units: [512,512]
      Activation functions: ['relu','relu','softmax']
      Learning rate: 0.003
      Batch size: 100
      No. of epochs: 32
      Optimizer: Mini-batch Gradient Descent

Result:

![alt text](https://github.com/c0rtis-prime/cortopy/blob/master/results/mnist-Loss_plot_1.png "Loss plot") 
      
      Train accuracy = 92.691667%
      Test accuracy = 92.65% 

#### Configuration-2:
      No. of hidden  layers: 2
      No. of hidden units: [512,512]
      Activation functions: ['relu','relu','softmax']
      Learning rate: 0.03
      Batch size: 100
      No. of epochs: 32
      Optimizer: Gradient Descent with momentum

Result:

![alt text](https://github.com/c0rtis-prime/cortopy/blob/master/results/mnist-Loss_plot_2.png "Loss plot") 
      
      Train accuracy = 95.71%
      Test accuracy = 95.23%

