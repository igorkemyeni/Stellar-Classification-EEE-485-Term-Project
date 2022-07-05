"""
Anıl Bayram Göğebakan
İsmail Görkem Yeni
EEE485-Project
KNN Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt


train_set = np.load(r'D:\GitHub\Stellar-Classification\train_set.npy')
test_set = np.load(r'D:\GitHub\Stellar-Classification\test_set.npy')
validation_set = np.load(r'D:\GitHub\Stellar-Classification\validation_set.npy')


def knn(k,train_set, test_set):
    
    conf_arr = np.zeros((3,3))

    train_label = train_set[:,-1]
    train_feat = train_set[:,:-1]
    
    test_label = test_set[:,-1]
    test_feat = test_set[:,:-1]
   
    for i in range(len(test_feat)):
        euc = np.linalg.norm(test_feat[i]-train_feat, axis=1)
        dist_w_label = np.hstack((np.expand_dims(euc, axis=1), np.expand_dims(train_label, axis=1)))
        a = dist_w_label[dist_w_label[:, 0].argsort()]
        
        maxlist = []
        count0 = 0
        count1 = 0
        count2 = 0
        for m in range(k):
            if a[m][1] == 0:
                count0 += 1
            elif a[m][1] == 0.5:
                count1 += 1
            else:
                count2 += 1
        maxlist.append(count0)
        maxlist.append(count1)
        maxlist.append(count2)
        
        output = maxlist.index(max(maxlist))
        
        conf_arr[output][int(test_label[i]*2)] += 1
        
    
        if i+1 % 300 == 0:
            print(str(i+1)+ " data points are calculated")
    print(str(i+1)+ " data points are calculated")
    print("Calculation has finished")
    return conf_arr

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

Accur = []
for i in range(1,11):
    
    conf_arr = knn(i,train_set, validation_set)
    acc = 100*np.trace(conf_arr)/conf_arr.sum()
    Accur.append(acc)
    
x = list(range(1,11))
plt.figure()
plt.plot(x,Accur)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN')


print("Best Accuracy: " + str(max(Accur)))
conf_arr = knn(3,train_set, test_set)

plot_conf(conf_arr)
