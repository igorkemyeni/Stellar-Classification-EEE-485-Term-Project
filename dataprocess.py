"""
Anıl Bayram Göğebakan
İsmail Görkem Yeni
EEE485-Project
Data Process Algorithm
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import seaborn as sns

def convert_one_hot(label_array):
    label_ohot = np.zeros((label_array.shape[0],3))
    
    for i in range(label_array.shape[0]):
        label_ohot[i][int(label_array[i])] = 1
        
    return label_ohot

random.seed(0)
start_time = time.time()

# file = open(r'D:\Gorkem\4-1\EE485\Proje\star_classification.csv')
file = open(r'D:\Github\Stellar-Classification\star_classification.csv')

csvreader = csv.reader(file)

header =  []
header = next(csvreader)

data =[]
for row in csvreader:
    data.append(row)
 
for i in range(len(data)):
    
    if data[i][13] == 'GALAXY':
        data[i][13] = 0
    elif data[i][13] == 'STAR':
        data[i][13] = 1
    else:
        data[i][13] = 2
    
    data[i] = [float(x) for x in data[i]]
    

data_arr= np.array(data)

c = data_arr[:,1:8]
d = data_arr[:,11].reshape(-1,1)
e = data_arr[:,14].reshape(-1,1)
class_label = data_arr[:,13].reshape(-1,1)
data_f = np.hstack((c,d,e,class_label))
      

data_f =np.delete(data_f, 79543, axis=0 )
data_normed = (data_f - data_f.min(axis=0)) / (data_f.max(axis=0)-data_f.min(axis=0))


mean, stdev = np.mean(data_normed, axis=0), np.std(data_normed, axis=0)

outliers = ((np.abs(data_normed[:,0] - mean[0]) > stdev[0])
            * (np.abs(data_normed[:,1] - mean[1]) > stdev[1])
            * (np.abs(data_normed[:,2] - mean[2]) > stdev[2])
            * (np.abs(data_normed[:,3] - mean[3]) > stdev[3])
            * (np.abs(data_normed[:,4] - mean[4]) > stdev[4])
            * (np.abs(data_normed[:,5] - mean[5]) > stdev[5])
            * (np.abs(data_normed[:,6] - mean[6]) > stdev[6])
            * (np.abs(data_normed[:,7] - mean[7]) > stdev[7])
            * (np.abs(data_normed[:,8] - mean[8]) > stdev[8]))

# extracting outliers from the dataset
no_outliers = data_normed[outliers == False]

           
#To show class imbalance.
count0 = 0
count1 = 0
count2 =0

for i in range(class_label.shape[0]):
    if class_label[i][0] == 0:
        count0 += 1
    elif class_label[i][0] == 1:
        count1 += 1
    else:
        count2 += 1
classnames = ('Galaxy', 'Star', 'Quasar')
counts = [count0, count1, count2]
y_pos = np.arange(len(classnames))

plt.bar(y_pos, counts, align= 'center', alpha = 0.4)
plt.xticks(y_pos,classnames)
plt.ylabel('Number')
plt.title('Number of Class Elements')
plt.show()
    
dataf1 = no_outliers[no_outliers[:,9]==0][:10000]
dataf2 = no_outliers[no_outliers[:,9]==0.5][:10000]
dataf3 = no_outliers[no_outliers[:,9]==1][:10000]

data_f = np.vstack((dataf1,dataf2,dataf3))

ind = random.sample(range(0,data_f.shape[0]),6000)
validation_set = data_f[ind]
data_f = np.delete(data_f, ind, axis = 0)

ind = random.sample(range(0,data_f.shape[0]),3000)
test_set = data_f[ind]
train_set = np.delete(data_f, ind, axis = 0)

print("--- %s seconds ---" % (time.time() - start_time))

np.save("Dataset.npy",data_normed)
np.save("Dataset2.npy",no_outliers)
np.save("train_set",train_set)
np.save("test_set",test_set)
np.save("validation_set",validation_set)

#%% Correlation Matrix

dataset_label = np.vstack((dataf1,dataf2,dataf3))[:,-1]*2
dataset_feat = np.vstack((dataf1,dataf2,dataf3))[:,:-1]
dataset_label = convert_one_hot(dataset_label)
all_data = np.hstack((dataset_feat, dataset_label))

df = pd.DataFrame(all_data, columns=["alpha", "delta", "u", "g", "r", "i", "z", "field_ID","redshift","Galaxy", "Star", "Quasar"])
plt.figure()
corr = df.corr().round(3)
corr.style.background_gradient(cmap='coolwarm')
sns.heatmap(df.corr(), annot=True, annot_kws={"fontsize":8})