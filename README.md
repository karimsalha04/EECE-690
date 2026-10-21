# EECE-690
# baseline code using the phase quantities 
# same is applied to the other set of features (sequence, delta phase, and delta sequence)
# sweeping to find best L2 regularization coeffecient
# plotting training vs. validation loss to choose the right number of iterations and prevent overfitting
# I am using the binary cross entropy loss 
# Step 1: run the data generation to create the data used/ Change directory to locate the excel file/ the code has 2 modes of operation you can start by the testing mode to check if everything is working well, then set test mode=false and generate all the data needed
# Step 2: use the baseline line code to train the model using different set of features 
# I split data 75% for training 10% for validation and 10 % for testing: the testing now is kept aside we need to check with the best regularization and iterations number, later on this test is going to be used with the training data because our testing is on a completely different network
