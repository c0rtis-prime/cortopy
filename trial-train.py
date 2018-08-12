import cortopy.models as models


hidden_units = [512,512]

classifier = models.dense_model( X_train.values, Y_train_enc.values,
                                 hidden_units,
                                 act_fn_list=['relu','relu','softmax'], 
                                 cost="softmax_cross_entropy_w_logits" )


'''
learning_rates = [1e-4, 9e-5, 7e-5,]

for lr in learning_rates:
    
    hidden_units = [512,512]
    classifier = models.dense_model( X_train.values, Y_train_enc.values,
                                     hidden_units,
                                     act_fn_list=['relu','relu','softmax'], 
                                     cost="softmax_cross_entropy_w_logits" )
    
    classifier.train( X_train.values, Y_train_enc.values,
                      X_test.values, Y_test_enc.values,
                      learning_rate=lr,     
                      batch_size=100,  
                      epochs = 20,
                      optimizer = "ADAM",
                      momentum_beta = 0.9,
                      rmsprop_beta = 0.999 )

'''

classifier.train( X_train.values, Y_train_enc.values,
                  X_test.values, Y_test_enc.values,
                  learning_rate=0.0001,     
                  batch_size=100,  
                  epochs = 52,
                  optimizer = "ADAM",
                  momentum_beta = 0.9,
                  rmsprop_beta = 0.999 )

classifier.save_weights("results/mnist-weights")