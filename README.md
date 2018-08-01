# cortopy
Deep Neural Network framework module built from scratch
---
##Currently available options:

*Classes:* 1. dense_model


*Activation functions:* 
                    1. ReLU
                    2. Sigmoid                         
                    3. Hyperbolic tangent                       
                    4. Softmax
                       
*Losses:*              

                    Mean squared error
                    Binary cross entropy
                    Softmax cross entropy (multiclass)
                       
*Optimizers:*      

                    Mini-batch Gradient Descent
                    
                    
##Usage in code:

*Import "models" and create a dense_model object:*
```python
import models
model = models.dense_model(X_train, Y_train_enc, hidden_units, act_fn_list, cost)
```
