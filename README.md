# cortopy
Deep Neural Network framework module built from scratch
---
##Currently available options:

*Classes:* 

                    dense_model


*Activation functions:* 

                    ReLU
                    Sigmoid                         
                    Hyperbolic tangent                       
                    Softmax
                       
*Losses:*              

                    Mean squared error
                    Binary cross entropy
                    Softmax cross entropy (multiclass)
                       
*Optimizers:*      

                    Mini-batch Gradient Descent
                    
                    
##Usage in code:

*Import "models" and create a dense_model object:*
```python
model = models.dense_model(X_train, Y_train_enc, hidden_units, act_fn_list, cost)
```
