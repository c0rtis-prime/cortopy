# cortopy
Deep Neural Network framework module built from scratch with numpy 

---

## Currently available options:

*Classes:*   

                    1. dense_model


*Activation functions:*

                    1. ReLU
                    2. Sigmoid                         
                    3. Hyperbolic tangent                       
                    4. Softmax
                       
*Losses:*              

                    1. Mean squared error
                    2. Binary cross entropy
                    3. Softmax cross entropy (multiclass)
                       
*Optimizers:*      

                    1. Mini-batch Gradient Descent
                      
---  

## Usage in code:

### Import "models" and create a dense_model object:
```python
import models
model = models.dense_model(X_train, Y_train_enc, hidden_units, act_fn_list, cost)
```

### Train your model:
```python
classifier.train(X_train, Y_train_enc,
                 X_test, Y_test_enc,
                 learning_rate,     
                 batch_size,  
                 epochs)
```
The training process returns the Loss v/s Epoch plot

### Make predictions:
```python
prediction = model.predict(X_sample)
```

### Save weights:
```python
model.save_weights(path)
```

### Load weights into the model:
```python
model = models.dense_model(X_train-like, Y_train_enc-like, hidden_units, act_fn_list, cost)
model.load_weights(path)
```

# MNIST handwritten digits dataset
This module was used to create a classifier for the MNIST dataset with the following settings:

      No. of hidden  layers: 2
      No. of hidden units: [512,512]
      Activation functions: ['relu','relu','softmax']
      Learning rate: 0.003
      Batch size: 100
      No. of epochs: 32

and produced the following results:

      ![alt text]("https://github.com/c0rtis/cortopy/blob/master/results/mnist-error plot.png" "Loss plot") 
      Train accuracy = 92.6917%
      Test accuracy = 92.65%
<<<<<<< HEAD

=======
>>>>>>> 0e3963c1bb9ea9dc485ef2dd1d48563e128d145a
