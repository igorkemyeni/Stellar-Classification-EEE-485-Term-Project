"""
Anıl Bayram Göğebakan
İsmail Görkem Yeni
EEE485-Project
Random Forest Algorithm
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import time

train_set = np.load(r'D:\GitHub\Stellar-Classification\train_set.npy')
test_set = np.load(r'D:\GitHub\Stellar-Classification\test_set.npy')
validation_set = np.load(r'D:\GitHub\Stellar-Classification\validation_set.npy')

def bagging(data, num_tree):
    

    sample_num = int(train_set.shape[0]/num_tree)
    tree_list = np.zeros((num_tree, sample_num, train_set.shape[1]))
    
    for i in range(num_tree):
        tree = random.choices(data, k = sample_num)
        tree_list[i] = tree
    return tree_list

def find_threshold(parent,feature):

    gp = gini(parent)   

    best_info = -999     
    for threshold in parent[:,feature]:
        left, right = get_split(parent, threshold, feature)

        gl = gini(left)
        gr = gini(right)
        
        ig = gp - (left.shape[0]/parent.shape[0])*gl -(right.shape[0]/parent.shape[0])*gr
        if ig > best_info:
            best_info = ig
            final_th = threshold
    return final_th

def go_left(index):
    return 2*index

def go_right(index):
    return 2*index+1
    
def get_split(data,threshold, feature):
    
    left = data[data[:,feature] <= threshold]
    right = data[data[:,feature] > threshold]
    
    return left, right

def find_category(child):
    cat, counts = np.unique(child[:,-1], return_counts=True)
    category = cat[np.argmax(counts)]
    accuracy = 100*max(counts)/np.sum(counts)
    return category, accuracy

def vote_result(results):
    category, counts = np.unique(results, return_counts=True)
    category = category[np.argmax(counts)]
    return category
    
def gini(data):
    count0 = 0
    counthalf = 0
    count1= 0

    if data.shape[0] == 0: 
        giniind = 1
    else:
        count0 = np.count_nonzero(data[:,-1]==0)
        counthalf = np.count_nonzero(data[:,-1]==0.5)
        count1= np.count_nonzero(data[:,-1]==1)
        giniind = 1 - ((count0/data.shape[0])**2 + (counthalf/data.shape[0])**2 + (count1/data.shape[0])**2)
                
    return giniind
    
def train(data, max_feat, num_tree):
    start = time.time()
    init_parent_list= bagging(data,num_tree)    
    
    # threshold list composed of threshold dictionaries for every subtree
    threshold_list = []
    # category list for all subtrees, composed of category_list for each tree
    final_category_list = []
    features_list = []

    terminals = []
    terminal_category = []
    
    for tree_ind in range(num_tree):
        # This list contains parents for every level. Size is updated for every level
        random_features = random.sample(range(2,train_set[0].size-1), max_feat)
        
        parent_list = [init_parent_list[tree_ind]] #initial update
        threshold_dict = {}
        category_list = [] # Category list for individual tree
        terminal_node_list = [] #terminals will be stored here as tuples
        terminal_category_list = []
        # print()
        # print("The features which are used in tree " + str(random_features))
        features_list.append(random_features)
        # print(str(tree_ind+1)+". tree is establishing.")
        level_ind = 0
        
        for feat in random_features:
            child_list = []
            level_threshold = []
            coordinate = 0
            for parent in parent_list: 
                
                if parent is None:
                    left = None
                    right = None
                    th = 0
                    coordinate += 2
                else:
                    # Determine threshold and append it to level_threshold                
                    th = find_threshold(parent, feat)
                    left, right = get_split(parent,th, feat)
                    
                    if gini(left) == 0:
                        terminal_node_list.append((level_ind+1,coordinate))
                        category, _ = find_category(left) # _ to supress the output
                        terminal_category_list.append(category)
                        left = None                 
                    coordinate += 1
                        
                    if gini(right) == 0:
                        terminal_node_list.append((level_ind+1,coordinate))
                        category, _ = find_category(right) # _ to supress the output
                        terminal_category_list.append(category)
                        right = None                    
                    coordinate += 1
                
                child_list.append(left)
                child_list.append(right)
                level_threshold.append(th)
          
            threshold_dict[level_ind] = level_threshold                
                
            # Going next level, refresh generation
            parent_list = child_list
               
            
            # print(str(level_ind+1)+". level thresholds are determined.")
            level_ind += 1
        
        terminals.append(terminal_node_list)
        terminal_category.append(terminal_category_list)
        threshold_list.append(threshold_dict)
        
        # Find classes of each child
        for child in parent_list:
            if child is not None:
                category, accuracy = find_category(child)
                category_list.append(category)
                # print(accuracy)
            else:
                category_list.append(None)
                # print("None")
        
        final_category_list.append(category_list)
    print("Time spend while training: {:.2f}".format(time.time() - start))
            
    return threshold_list, features_list, final_category_list, terminals, terminal_category

def predict(inp, threshold_list, feat_list, category_list, terminals, terminal_category): 
    
    predictions = []
    for tree in range(len(threshold_list)): # tree = index of tree
        horizontal_index = 0
        for level in range(len(feat_list[tree])+1): # level = level index in tree
            
            if (level,horizontal_index) in terminals[tree]: # check that node is terminal
                term_ind = terminals[tree].index((level,horizontal_index))
                predictions.append(terminal_category[tree][term_ind])
                break
            
            elif level == len(feat_list[tree]): # check that node is bottom terminal
                predictions.append(category_list[tree][horizontal_index])
                break
            
            else:
                feature = feat_list[tree][level]
                th = threshold_list[tree][level][horizontal_index]
                if inp[feature] < th:
                    horizontal_index = go_left(horizontal_index)
                    
                else:
                    horizontal_index = go_right(horizontal_index)
    
    return predictions

def plant_forest(max_feat, num_tree, train_set=train_set, validation_set=validation_set): 
    acc_list = []
    conf_arr = np.zeros((3,3))
    for i in range(1):    
        threshold_list, feat_list, category_list, terminals, terminal_category = train(train_set, max_feat=3, num_tree=40)
        
        acc = 0
        for inp in validation_set:
            result_list = predict(inp, threshold_list, feat_list, category_list, terminals, terminal_category)
            result = vote_result(result_list)
            # print(result)
            # print(inp[-1])
            conf_arr[int(result*2)][int(inp[-1]*2)] += 1
            
            if inp[-1] == result:
                acc += 1
        acc_list.append(100*acc/validation_set.shape[0])
    average_accuracy = sum(acc_list)/len(acc_list)
    print("Max feature: {} Number of tree: {} Accuracy value: {:.2f}".format(max_feat, num_tree,average_accuracy))
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
        
# conf_arr = plant_forest(max_feat=5,num_tree=40, train_set=train_set, validation_set=test_set)
# plot_conf(conf_arr)


plant_forest(max_feat=9, num_tree=80)    
print()
plant_forest(max_feat=8, num_tree=80)
print()
plant_forest(max_feat=7, num_tree=80)
print()
# plant_forest(max_feat=6, num_tree=80)
# plant_forest(max_feat=5, num_tree=80)
# plant_forest(max_feat=4, num_tree=80)
# plant_forest(max_feat=3, num_tree=80)
# plant_forest(max_feat=2, num_tree=80)

# print()
plant_forest(max_feat=9, num_tree=40)  
print()  
plant_forest(max_feat=8, num_tree=40)
print()
plant_forest(max_feat=7, num_tree=40)
print()
# plant_forest(max_feat=6, num_tree=40)
# plant_forest(max_feat=5, num_tree=40)
# plant_forest(max_feat=4, num_tree=40)
# plant_forest(max_feat=3, num_tree=40)
# plant_forest(max_feat=2, num_tree=40)

# print()
plant_forest(max_feat=9, num_tree=20)
print()
plant_forest(max_feat=8, num_tree=20)
print()
plant_forest(max_feat=7, num_tree=20)
print()
# plant_forest(max_feat=6, num_tree=20)
# plant_forest(max_feat=5, num_tree=20)
# plant_forest(max_feat=4, num_tree=20)
# plant_forest(max_feat=3, num_tree=20)
# plant_forest(max_feat=2, num_tree=20)




    