"""
Anıl Bayram Göğebakan
İsmail Görkem Yeni
EEE485-Project
2 Layer Neural Networks Algorithm
"""

#%% Imported libraries and some local functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import time

np.random.seed(0)

start_time = time.time()
# random.seed(1)
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def convert_one_hot(label_array):
    label_ohot = np.zeros((label_array.shape[0],3))
    
    for i in range(label_array.shape[0]):
        label_ohot[i][int(label_array[i])] = 1
        
    return label_ohot

def create_batch(dataset, batch_size):
    # input: dataset
    ## This function takes dataset and give random indexes in de
    # output: np.array(number of indexes, index values)
    ind = np.arange(dataset.shape[0])
    batch_matrix = np.random.choice(ind, (dataset.shape[0]//batch_size,batch_size), replace=False)
    return batch_matrix


def d_tanh(x):
    return 1-np.tanh(x)**2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def d_softmax(x):
    # return softmax(x)*(1-softmax(x))
    return x

def cross_ent_loss(label,predicted):
    # return np.sum(-label*np.log(predicted))
    # return 1
    return -np.sum(label*np.log(predicted))/predicted.shape[0]

def two_layer_forw_pass(inp, W_in, W_m, W_out):
    first_layer_in_raw = np.transpose(W_in) @ inp
    first_layer_in_raw = np.expand_dims(first_layer_in_raw, axis=1)
                
    # W_in = 10xn, train_set[ind] = 10x1
    first_layer_in_act = np.tanh(first_layer_in_raw) 
                 
    # first_out = nx1
    second_layer_in = np.concatenate((first_layer_in_act, np.array([[-1]])))
    # second_in = n+1x1
    second_layer_in_raw = np.transpose(W_m) @ second_layer_in
    # W_out = 11x1, first_out = n+1x1
    second_layer_in_act = np.tanh(second_layer_in_raw)                  
    # final_out = 1x1
    third_layer_in = np.concatenate((second_layer_in_act, np.array([[-1]])))
    #
    final_out_raw = np.transpose(W_out) @ third_layer_in
    #
    final_out_act = softmax(final_out_raw)
    
    
    return final_out_act


#%% Data Preprocess
train_set = np.load(r'C:\Users\MONSTER\Documents\GitHub\Stellar-Classification\train_set.npy')
test_set = np.load(r'C:\Users\MONSTER\Documents\GitHub\Stellar-Classification\test_set.npy')
validation_set = np.load(r'C:\Users\MONSTER\Documents\GitHub\Stellar-Classification\validation_set.npy')

train_label = train_set[:,-1]*2
train_feat = train_set[:,2:-1]



validation_label = validation_set[:,-1]*2
validation_feat = validation_set[:,2:-1]

#Test set to be used in testing purposes

test_label = test_set[:,-1]*2
test_feat = test_set[:,2:-1]
       
validation_label = convert_one_hot(validation_label)   
train_label = convert_one_hot(train_label)
test_label = convert_one_hot(test_label)
# df = pd.DataFrame(train_feat, columns=["alpha", "delta", "u", "g", "r", "i", "z", "field_ID","redshift"])
# plt.figure()
# corr = df.corr().round(3)
# corr.style.background_gradient(cmap='coolwarm')
# sns.heatmap(df.corr(), annot=True, annot_kws={"fontsize":8})


#%%
### ----------- PARAMETERS -----------
n = 30 # Number of neurons in the hidden layer 30
p = 10 # Number of neurons in the second hidden layer 10
mu, sigma = 0, 0.3 # mean and standard deviation 0.3
#initialized bias term
batch_size = 20
learn_rate = 0.001 # 0.001
epoch_num = 40
momentum = 0.1




#%%
def NeuralNetwork(train_label, test_label, validation_label,train_feat, test_feat, validation_feat, batch_size,epoch_num, n,p, learn_rate, momentum):

    
    ### ----------- PARAMETERS -----------
    
    #Random weights are initialized for input layer
    # train_feat.shape[1] = 9 which is the total feature number
    w_in = np.random.normal(mu, sigma, (train_feat.shape[1]+1, n))
    
    #Random weights are initialized for outputlayer
    w_m = np.random.normal(mu, sigma,(n+1,p))
    
    #Random weights are initialized for outputlayer
    w_out = np.random.normal(mu, sigma,(p+1,3))
    bias = [-1] 
    #Train set is expanded and bias term is added
    Bias_vec = np.full((train_feat.shape[0],1), bias)
    train_feat = np.concatenate((train_feat, Bias_vec), axis=1)
    
    #Validation set is expanded and bias term is added
    Bias_vec = np.full((validation_feat.shape[0],1), bias)
    validation_feat = np.concatenate((validation_feat, Bias_vec), axis=1)
    
    
    #Test set is expanded and bias term is added
    Bias_vec = np.full((test_feat.shape[0],1), bias)
    test_feat = np.concatenate((test_feat, Bias_vec), axis=1)
    
    
    error_ar = np.zeros(epoch_num)
    loss_ar = np.zeros((epoch_num))
    error_validation_ar = np.zeros(epoch_num)
    acc_ar = np.zeros(epoch_num)
    acc_validation_ar = np.zeros(epoch_num)
    bad_acc_ar = np.zeros(epoch_num)
    bad_acc_validation_ar = np.zeros(epoch_num)
    for i in range(epoch_num):
        error= 0
        loss = 0
        acc = 0
        acc_validation = 0
        error_validation = 0
        bad_acc = 0
        bad_acc_validation = 0
        batches = create_batch(train_feat, batch_size)
        change1 = 0
        change2= 0
        change3 = 0
        for bat in batches: # bat is one batch
            dW_out = 0
            dW_in = 0
            dW_m = 0
            
            for ind in bat: # ind is index of input in dataset
    
                # Forward Pass
                # out = two_layer_forw_pass(train_feat[ind], w_in, w_m, w_out)
                
                first_layer_in_raw = np.transpose(w_in) @ np.transpose(train_feat[ind:ind+1])
                
                # W_in = 10xn, train_set[ind] = 10x1
                first_layer_in_act = np.tanh(first_layer_in_raw) 
                             
                # first_out = nx1
                second_layer_in = np.concatenate((first_layer_in_act, np.array([[-1]])))
                # second_in = n+1x1
                second_layer_in_raw = np.transpose(w_m) @ second_layer_in
                # W_out = 11x1, first_out = n+1x1
                second_layer_in_act = np.tanh(second_layer_in_raw)                  
                # final_out = 1x1
                third_layer_in = np.concatenate((second_layer_in_act, np.array([[-1]])))
                #
                final_out_raw = np.transpose(w_out) @ third_layer_in
                #
                final_out_act = softmax(final_out_raw)
                out = final_out_act
                
                # Change error function
                # expanded_label = np.expand_dims(train_label[ind], axis=1)
                label = train_label[ind:ind+1].T 
                e = label - final_out_act
                # e = train_label[ind:ind+1].T  - out
                # e = train_label[ind:ind+1].T * e
                # e = e**2
                err = np.sum(np.transpose(train_label[ind:ind+1]) * e)
                
                loss += cross_ent_loss(train_label[ind:ind+1], out) / (batches.shape[0]*batches.shape[1])
                error += err / (batches.shape[0]*batches.shape[1])
            
                
        
                # For W_out, chain of dependence is as following
                #    E -> e -> o -> v -> w
                
    
                sq_final_out = np.identity(3)
                np.fill_diagonal(sq_final_out, final_out_raw)
                
                # grad_out = d_softmax(sq_final_out) @ e
                
                grad_out = e
                dW_out +=  (grad_out @ np.transpose(third_layer_in))/batch_size
                
        
                # For W_in, chain of dependence is as following
                # E -> e -> o -> v -> y -> $v -> $w
                
                sq_second_out = np.identity(p)
                np.fill_diagonal(sq_second_out, second_layer_in_raw)
                
                grad_m = d_tanh(sq_second_out) @ w_out[:-1] @ grad_out
                dW_m += (grad_m @ np.transpose(second_layer_in))/batch_size
                
                
                sq_first_out= np.identity(n)
                np.fill_diagonal(sq_first_out, first_layer_in_raw)
                
                grad_in = d_tanh(sq_first_out) @ w_m[:-1] @ grad_m
                dW_in = (grad_in @ train_feat[ind:ind+1])/ batch_size
                
                if np.transpose(train_label[ind:ind+1])[np.argmax(final_out_act)] == 1 and not np.isnan(final_out_act[np.argmax(final_out_act)]):
                    acc += 1
                    if np.argmax(final_out_act) == 0:
                        bad_acc += 1
                
            new_change1 = learn_rate*np.transpose(dW_out) + momentum*change1
            w_out += new_change1# - 0.01*np.sum(w_out)
            
            new_change2 =learn_rate*np.transpose(dW_m) + momentum*change2
            w_m += new_change2# - 0.01*np.sum(w_m)
            
            new_change3 = learn_rate*np.transpose(dW_in) + momentum*change3
            w_in += new_change3# - 0.01*np.sum(w_in)
            
            change1 = new_change1
            change2 = new_change2
            change3 = new_change3
            
            
    
    
        
        for k in range(len(validation_feat)):
            validation_out = two_layer_forw_pass(validation_feat[k], w_in, w_m, w_out)
            e = validation_label[k:k+1].T  - validation_out
            err = np.sum(np.transpose(validation_label[k:k+1]) * e)
            
            error_validation += err/len(validation_feat)
            
            if np.transpose(validation_label[k:k+1])[np.argmax(validation_out)] == 1 and not np.isnan(validation_out[np.argmax(validation_out)]):
                acc_validation += 1
                
                if np.argmax(validation_out) == 0:
                    bad_acc_validation += 1
                
        error_ar[i] = error
        loss_ar[i] = loss
        acc_ar[i] = 100*acc/(batches.shape[0]*batches.shape[1])     
        error_validation_ar[i] = error_validation
        acc_validation_ar[i] = (100*acc_validation)/len(validation_feat)
        
        bad_acc_ar[i] = 100*bad_acc/(batches.shape[0]*batches.shape[1])
        bad_acc_validation_ar[i] = (100*bad_acc_validation)/len(validation_feat)
    
    
            
            
        if i % 1 == 0:
            
            print("Train loss {:.2f}, after {} epoch.".format(float(loss), i+1)) 
            print("Train error  {:.2f} after {} epoch.".format(float(error), i+1)) 
            print("Train accuracy  {:.2f} after {} epoch.".format(float(acc_ar[i]), i+1)) 
            # print("Ratio of first accuracy  {:.2f} after {} epoch.".format(float(bad_acc_ar[i]), i+1)) 
            
            print("Validation error  {:.2f} after {} epoch.".format(float(error_validation), i+1)) 
            print("Validation accuracy  {:.2f} after {} epoch.".format(float(acc_validation_ar[i]), i+1)) 
            # print("Ratio of first Test accuracy  {:.2f} after {} epoch.".format(float(bad_acc_test_ar[i]), i+1)) 
    
            print()
    fig, (ax1, ax2) = plt.subplots(1,2,constrained_layout=True)  # Create a figure and an axes.
    
    ax1.plot(error_ar, label='train')  # Plot some data on the axes.
    ax1.plot(error_validation_ar, label='validation')  # Plot more data on the axes...
    ax1.set_ylabel('Loss value')  # Add an x-label to the axes.
    ax1.set_xlabel('Epoch Number')
    ax1.set_title("Loss values")  
    ax1.legend()
    
    
    ax2.plot(acc_ar, label='train')  # ... and some more.
    ax2.plot(acc_validation_ar, label='validation')
    ax2.set_ylabel('Accuracy value')
    ax2.set_xlabel('Epoch number')# Add a y-label to the axes.
    ax2.set_title("Accuracy values ")  # Add a title to the axes.
    ax2.legend()  # Add a legend
    
    fig.show()
    
    #find the final test score of the network
    def test(w_in, w_m, w_out, test_feat,test_label):
        error_test = 0
        acc_test = 0
        bad_acc_test = 0
        conf_arr = np.zeros([3,3])
        
        for k in range(len(test_feat)):
            test_out = two_layer_forw_pass(test_feat[k], w_in, w_m, w_out)
            e = test_label[k:k+1].T  - test_out
            err = np.sum(np.transpose(test_label[k:k+1]) * e)
            test_ind = np.argmax(test_out)
            lbl_ind = np.argmax(test_label[k:k+1])
            conf_arr[test_ind][lbl_ind] += 1
            
            error_test += err/len(test_feat)
            
            if np.transpose(test_label[k:k+1])[np.argmax(test_out)] == 1 and not np.isnan(test_out[np.argmax(test_out)]):
                acc_test += 1
                
                if np.argmax(test_out) == 0:
                    bad_acc_test += 1
        acc_test_f = (100*acc_test)/len(test_feat)
        
        
    
        print("Test accuracy  {:.2f} of the model.".format(float(acc_test_f), 1))
        return acc_test_f, conf_arr
    
    
    ret_acc, conf_arr = test(w_in, w_m, w_out, test_feat, test_label)
    
    return ret_acc, conf_arr




#Three diff batch_size, first layer neurons n = 30, second_layer neurons p = 10, learn_rate = 0.001, momentum = 0
# ret1, conf_arr=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 20,epoch_num= 40, n= 30,p=10, learn_rate= 0.001, momentum= 0.1)
ret4, conf_arr=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 20,epoch_num= 40, n= 30,p=10, learn_rate= 0.001, momentum= 0)
# ret2=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 20,epoch_num= 40, n= 30,p=10, learn_rate= 0.001, momentum= 0)
# ret3=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 50,epoch_num= 50, n= 30,p=10, learn_rate= 0.001, momentum= 0)


# ret5=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 2,epoch_num= 50, n= 30,p=10, learn_rate= 0.001, momentum= 0.05)
# ret6=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 3,epoch_num= 50, n= 30,p=10, learn_rate= 0.001, momentum= 0.05)

# ret7=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 1,epoch_num= 50, n= 6,p=2, learn_rate= 0.001, momentum= 0)
# ret8=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 2,epoch_num= 50, n= 6,p=2, learn_rate= 0.001, momentum= 0)
# ret9=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 3,epoch_num= 50, n= 6,p=2, learn_rate= 0.001, momentum= 0)

# ret10=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 1,epoch_num= 50, n= 15,p=4, learn_rate= 0.01, momentum= 0)
# ret11=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 1,epoch_num= 50, n= 15,p=4, learn_rate= 0.001, momentum= 0)

# ret11=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 2,epoch_num= 50, n= 15,p=4, learn_rate= 0.01, momentum= 0)
# ret12=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 2,epoch_num= 50, n= 15,p=4, learn_rate= 0.001, momentum= 0)

# ret13=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat,  batch_size = 3,epoch_num= 50, n= 15,p=4, learn_rate= 0.01, momentum= 0)
# ret14=NeuralNetwork(train_label,test_label,validation_label,train_feat, test_feat, validation_feat, batch_size = 3,epoch_num= 50, n= 15,p=4, learn_rate= 0.001, momentum= 0)

#Plot the confusion matrix for the test results
def plot_conf(conf_arr):
    label = ["Galaxy", "Star", "Quasar"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_arr, interpolation='nearest', cmap='PiYG')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+label)
    ax.set_yticklabels(['']+label)
    ax.set_title('Labels')
    ax.set_ylabel('Outputs')
    
    for (x, y), value in np.ndenumerate(conf_arr):
        plt.text(x, y, int(value), va="center", ha="center")

plot_conf(conf_arr)

print("--- %s seconds ---" % (time.time() - start_time))
