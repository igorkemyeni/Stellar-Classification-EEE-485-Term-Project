[link1]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig1.jpg
[link2]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig2.png
[link3]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig3.jpg
[link3.5]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig3.5.jpg
[link4]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig4.jpg
[link5]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.png
[link5.5]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.5.jpg
[link5.75]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.75.jpg
[link5.8]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.8.jpg
[link5.90]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.90.jpg
[link5.91]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.91.jpg
[link5.92]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.92.jpg
[link5.93]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.93.jpg
[link5.94]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig5.94.jpg
[link6]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig6.jpg
[link7]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig7.jpg
[link8]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig8.jpg
[link9]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig9.jpg
[link9.1]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig9.1.jpg
[link9.2]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig9.2.jpg
[link10]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig10.jpg
[link11]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Fig11.jpg
[link_table1]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Table1.jpg
[link_table2]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Table2.jpg
[link_table3]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Table3.jpg
[link_table4]: https://github.com/anilbayramgogebakan/Stellar-Classification/blob/main/src/Table4.jpg

# Stellar Classification by Using Different Machine Learning Techniques

*Anıl Bayram Göğebakan, İsmail Görkem Yeni*

## 1. Abstract

 Recently, space agencies such as European Space Agency, NASA, etc. has accomplished one of the most difficult and challenging space missions which is sending and orbiting properly the space telescope James Webb. As of today, the space craft is successfully got into its orbit which will be functional completely in few months and will be able to send earth many images from the deep space. Therefore, there will be a vast amount of new space objects such as stars, galaxies, and quasars that are required to be classified. Hence, we thought that the need for a machine learning algorithm properly classifying these unknown space objects is inevitable. To manage the big-data coming from the telescope, we had to use a database including many data points with a good validity of the data.

## 2. Data Preprocessing

 As it was stated in the proposal, dataset is found from Kaggle database system. The [link](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) can be followed for the data  . This data is published by Sloan Digital Sky Survey and a ground telescope stationed in New Mexico, US. The dataset consists of one hundred thousand data points which is beneficial considering the possible vast amount of data which will be gathered from James Webb Telescope.

1. alpha = Right Ascension angle (at J2000 epoch)
2. delta = Declination angle (at J2000 epoch)
3. u = Ultraviolet filter in the photometric system
4. g = Green filter in the photometric system
5. r = Red filter in the photometric system
6. i = Near Infrared filter in the photometric system
7. z = Infrared filter in the photometric system
8. redshift = redshift value based on the increase in wavelength
9. fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation
10. cam_col = Camera column to identify the scanline within the run
11. run_ID = Run Number used to identify the specific scan
12. rereun_ID = Rerun Number to specify how the image was processed
13. field_ID = Field number to identify each field
14. spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
15. plate = plate ID, identifies each plate in SDSS
16. MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
17. obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS

 However, we decided to not use most of the ID features which are not changing, such as spec_obj_ID, or unique numbers such as obj_ID. In the data preprocessing step, we chose manually some features to use throughout the project, we believe these are the proper features for the project because other ones are generally ID features. Used features: 1 to 8, which are the features that are used to identify the class of the space object, and field ID which differs for space objects but also has repeating instances which we thought may change the results.

 In our dataset, there are three different classes which are ‘Galaxy’, ‘Star’, and ‘Quasar’. In order to use them properly in the coding phase, we changed the string names for the classes by 0, 1, and 2 respectively.

 ![Figure-1][link1]
<p align = "center">
Figure 1: Class Imbalance in the Dataset
</p>
  
 As you can see from the database there is class imbalance problem. We would expect the database to include 33.3% of data instances for each class however, there is huge dominancy in the number of Galaxy class. Hence, it is more difficult to distinguish minority classes from the majority class. In order to prevent this class imbalance problem, we gathered equal number of data instances from each of the classes which is 10000 for each class and worked with 30000 data points in total.

  ![Figure-2][link2]
<p align = "center">
Figure 2: Correlation Matrix of all Dataset
</p>

 As it can be observed from the correlation matrix, some features are correlated closely which are close to each other. That means there exist positive correlation such as if feature A increases, feature B increases as well. However, we can examine some negative correlation between features such as for ‘alpha’ and ‘z’ or ‘alpha’ and ‘field_ID’. That means that when one of them increases, other one decreases. The strong positive correlation between some features that is around 0.9 to 1 can cause problem that called ‘Multicollinearity’. This may occur when one predictor variable in a multiple regression model can be linearly predicted from the others with good accuracy.[1] However, we do not use logistic regression classifier in our project and moreover, decision trees are immune algorithms to such problems by its nature. Since we will use random forests, this problem won’t cause much problem for us neither when we use neural networks and KNN. Even though we said in the first report that we might use PCA for the final phase, we did not see it necessary because we could get high accuracies from all of our models around 80-to 90%.

 Since we have three different classes, one hot coded the categories and draw the correlation matrix accordingly. By that, we have the chance to see the feature correlation with specific classes. We also saw it necessary to not use the first two features in the dataset to implement in the methods because they have low correlation and therefore, decreased the performance of the models immensely.

 **Train, Test and Validation Sets**

 Since our data is quite big, we consulted to our TA and after having his confirmation, we did not do cross validation. Hence, we only decided to use 3 different sets in our models. These are train, validation and test sets. To achieve that, we separated the data into 3 different sized arrays randomly but the rates of all of the classes are very similar to each other, in other words, number of data points from different classes are almost the same. For the train, validation and test sets, we used 21k, 6k, and 3k data instances respectively, which is the approximate percentage values for the split in the industry [2].

 For the Neural Network Model, we used in our project, we used one-hot-encoding for the data labels. In other words, we changed integer 0, 1, and 2 to [1 0 0], [0, 1, 0], and [0, 0, 1] arrays.

 In total, our data preprocessing step runs fast and smooth and we set the data as we can use comfortably. It took around 1.3seconds to prepare our dataset properly.

## 3. Methods Used in the Project

### 3.1 k-Nearest Neighbors (KNN)

 KNN algorithm is an easy to implement supervised machine learning algorithm for both regression and classification tasks. However, its easiness to implement comes with a significant drawback, which is becoming slow to run when the data size increases.

 Working principle of the KNN algorithm is to find the distances between the data points and classify the data points according to the other k many closest data point which has the majority of the classes. For example, if 10 closest data points are made up by 5 stars, 3 galaxies and 2 quasars, that data point is classified as star [3]. Whenever the K value of our model decreases, the stability of the model decreases as well and results poorly.

 In order to properly design the model, we used Euclidian distance formula which has a simple logic as below.

   ![Figure-3][link3]
<p align = "center">
Figure 3: Euclidian Distance Formula Illustration [4]
</p>

 As we can see from the formula, we behave our data features as points in space and find the closest datapoints to one point. General formula of this is the following:

   ![Figure-3.5][link3.5]

 We have chosen our K value after few trials with different K values as it can be seen in the below figure. Since we had the most accuracy at K = 3, our optimal K value is 3.

   ![Figure-4][link4]
 <p align = "center">

Figure 4: Validation Accuracy with KNN Algorithm with Different K values.
</p>

 After the first report phase, in order to optimize our code, we added our validation set to the model. The validation accuracy is higher than the initial results we acquired in the first report phase. Around 90%. The final test accuracy is 90.9% for the KNN method.

   ![Figure-5][link5]

 <p align = "center">

 Figure 5: Confusion Matrix for the KNN Method
 </p>

### 3.2 Fully Connected Neural Network

 Our purpose in this project is to design models that are capable of classify multi-class dataset. One of the best algorithms to achieve this objective is to use Neural Networks. Neural networks are one of the foundation stones of machine learning algorithms, it is more complicated to implement compared to KNN algorithm, however, it gives more stable and accurate results in the end. For large dataset as we have here, it is faster and more reliable. We chose cross-entropy as our loss function as the following in our project.

   ![Figure-5.5][link5.5]

 Here, 𝑦𝑖’s are our true labels and 𝑦̂𝑖’s are the predicted results of the model. In order to compute the loss function, we need weights in the neural network and forward pass these weights to have the predicted outcome. Since its fast and commonly used, we used mini-batch stochastic gradient descent in our model as the learning algorithm. Before training at each epoch, the train set is shuffled and then the model is trained for all of the batches one by one and within each batch, the weights are updated. This is continued till the loss converges to a point. In order to update the weights of the network, the following gradient descent formula is used.

  ![Figure-5.75][link5.75]

 To calculate the above gradient descent formula, we need to successfully do forward pass and then backpropagation.

#### 3.2.1 Forward Pass

In our neural network model, we used 1 input, 2 hidden and 1 output layers. We used multi-layer neural network to have more accurate, complex and challenging model in our project. For the hidden layers, we used hyperbolic tangent activation function (tanh) and SoftMax in the output layer since our dataset has 3 different classes, if we would have been doing binary classification, sigmoid or tanh would be appropriate again. Below are the formulas for the activation functions.

![Figure-5.80][link5.80]

The results of the SoftMax function are the probability of the class i for that specific inputted data x. We have three different weight matrices, from input to 1st hidden layer, 1st to 2nd hidden layer and 2nd to output layer. General matrix representation of these weights are as the following, weights for biases are added to the last column.

![Figure-5.90][link5.90]

#### 3.2.2 Backpropagation

In order to properly update the weights, we used backward propagation algorithm. For the backpropagation, we find the local gradients for each neuron and the general demonstration of this algorithm is the following.

For delta values 

![Figure-5.91][link5.91]

In the delta formulas, Γ′(𝑣𝑜) stands for the derivative of the activation function of that layer (here output) and we update the weights according to these delta functions.

![Figure-5.92][link5.92]

This method continues till the epoch is ended. For the first delta value, we have to calculate the derivative of the SoftMax function which is difficult and complicated to calculate, therefore, we simply equated the first delta value to error function, which is cross-entropy loss function in our case. Therefore, we need to find the partial derivative of error E with respect to activation potential 𝑧𝑗 such that:

![Figure-5.93][link5.93]

#### 3.2.3 Gradient Descent Optimization

In adding to the first report, we had the need to optimize our gradient descent algorithm. One of the most popular and consistent ways of doing this task is using momentum. One of the drawbacks of the gradient descent algorithm is that it can move upwards when it is going downhill towards the minima due to gradient of some points. To prevent this error, we can add the history of the past gradients when updating our weights. This phenomenon is called momentum as in the physics similar to acceleration in a direction.

The formula for the Stochastic Gradient Descent with Momentum is as the following [4].

![Figure-5.93][link5.93]

#### 3.2.4 Results

In our final implementation for the two-layer neural network. We acquired the results in following graphs. One of the results is obtained using momentum constant 0.1, other with zero momentum. After many trials, we acquired our final parameters as in the below table.

![Table-1][link_table1]
<p align = "center">
Table 1. Networks with Different Parameters with Epoch Num = 20
</p>

![Table-2][link_table2]
<p align = "center">
Table 2. Table of Parameters for Neural Network
</p>

From Table 1, we see that there are good results like 92% accuracy. However, these results were not stable than the optimal result found with the parameters in Table 2. Nevertheless, observing and testing the effect of different parameters for a neural network is always beneficial.

![Figure-6][link6]
<p align = "center">
Fig 6. Loss and Accuracy Values Through the Epochs for both Train and Validation Datasets without Momentum.
</p>

All of the training and validation process lasts approximately 109 seconds which was expected for two layers neural network with so many matrix multiplication calculations in it.

![Figure-7][link7]
<p align = "center">
Fig 7. Loss and Accuracy Values Through the Epochs for both Train and Validation Datasets with Momentum.
</p>

The test accuracy for the network we used momentum is 88% while the test accuracy for the network without momentum, the accuracy was 89.57%. Although we expected the contrary, we believe the results were consistent in itself.

![Figure-8][link8]
<p align = "center">
Fig 8. Confusion Matrix for the Neural Network
</p>

As it can be also proven from the confusion matrix, we have almost found all of the classes successfully. The number of false positives or negatives are low and true positives are high.

### 3.3 Random Forest

As our final model, we have implemented Random Forest. Random Forest classifier consist of large number of decision trees in it. These decision trees work with a simple logic as it is seen in the below figure. If the feature of the data instance satisfies the condition on the branch of the tree, it follows that branch as next step, and stops when there is no feature left to investigate.

![Figure-9][link9]
<p align = "center">
Fig 9. Decision Tree Illustration [6]
</p>

Compared to single decision tree, random forest model outperforms decision tree because of the collaboration of kindly comparatively uncorrelated decision trees. Lower the correlation between these individual trees in the random forest, model performs better because they protect themselves from their individual errors. To ensure that the correlation between decision trees is not high, there exist two different methods, which are bagging and chosing features randomly to be compared in decision trees. We implemented both of these methods in our model in order to have best possible random forest which can even identify uncorrelated data instances.

Bagging method is the method where we chose different training set with small changes, since these small changes can alter the decision trees strictly, random forest take the advantage of this phenomena by allowing each tree to randomly sample.

The other method is to chose features randomly to be compared in decision trees. This even further decreases the correlation between different decision trees and results in more diversification.

In our implementation of the code, we initially focused on generating a single tree. To do that, we have to split the data we are working on by a threshold. These threshold values are found by looking at the Gini values of our initial complete data, and the Gini values of the data we are going to split. For example, we are working with the ‘redshift’ feature, splitting the data to branches as of a tree, and we are taking the best threshold which gives us the optimal information gain.

Gini index is found by subtracting the sum of the squared probabilities of different classes in the data from 1 such that:

![Figure-9.1][link9.1]

Here C stands for the classes in the data which are star, galaxy and quasar. And 𝑝𝑖 is the probability of choosing the class i from the data of the branch. Then using the Gini values found for parent node, and its children nodes, the information gain is calculated. We want information gain to be always maximized when we are splitting a node, therefore, the final threshold we chose for that node is the one that gives the maximum information gain [7].

The formula for the Information Gain is as the following

![Figure-9.2][link9.2]

Here IG stands for Information gain, 𝐼(𝐷𝑃) represents the information of the parent node, in our case it is Gini of the parent node, 𝐼(𝐷𝑙𝑒𝑓𝑡) stands for the Gini of the left child node and 𝐼(𝐷𝑟𝑖𝑔ℎ𝑡) is the Gini of the right child node. 𝑁𝑝,𝑁𝑙𝑒𝑓𝑡,𝑁𝑟𝑖𝑔ℎ𝑡 represent the number of the element in parent, left child and right child respectively.

This branching process for the tree goes on until we reach out the maximum depth in the tree, this depth value is given as the maximum number of features we would like to use in our tree. If one node of the tree gives us elements all from the same class, it is understood that this node is terminal node. As an example, in the tree in Fig 6, there are three terminal nodes, two on the left bottom and one in the right upper node which has no child.

Our parameters in the random forest model are maximum depth = maximum features and number of trees in the forest.

#### Results

![Figure-10][link10]
<p align = "center">
Fig 10. Confusion Matrix for the Random Forest Model
</p>

We acquired a good accuracy in random forest after many trials, as it can be seen from the confusion matrix above. The highest final accuracy we achieved was around 91% for the above confusion matrix. But generally, we acquire accuracy around 80-90%. All of the random forest algorithm runs in approximately 2.3 seconds. This was quite faster than what we expected because of all of the sources we search through mentioned about the slowness of this method.

![Table-3][link_table3]
<p align = "center">
Table 3. Achieved accuracies from different Random Forests
</p>

## 4. Results

We believe all of our methods works properly considering the data we have and the complexities of different methods. As expected we spend not much time for the KNN method due to its simplicity, however, Neural Network method and Random Forests method took a lot of time while coding. The following table shows the results as a summary.

![Table-4][link_table4]
<p align = "center">
Table 4. Final Test Accuracy Table for the Methods
</p>

## 5. Conclusion

The purpose of this project is to predict the type of the stellar object from the data of the space telescope in the observatories or space such as James Webb Space Telescope. Firstly, the class imbalance problem is tried to be handled since almost 2/3 of the data is specified as ‘Galaxy’, then using the balanced dataset, we believe that the results we acquired are satisfactory.

We learned how to look a real-life problem in the scope of machine learning and successfully solve it by using different machine learning methods and models such as K-Nearest Neighbor, Multi-Layer Neural Network and Random Forest. We used some optimization methods to optimize our results in the methods.

#### Gantt Chart

![Figure-11][link11]
<p align = "center">
Fig 11. Gantt Chart that we Followed Throughout the Semester
</p>

## 6. References

[1] Badr, W. ‘Why Feature Correlation Matters… A Lot!’, towardsdatascience.com, Accessed on 9.04.2022 from https://towardsdatascience.com/why-feature-correlation-matters-a-lot- 847e8ba439c4

[2] Solawetz, J. ‘Train, Validation, Test Split for Machine Learning’, blog.roboflow.com. Accessed on 8.05.2022 from https://blog.roboflow.com/train-test-split/

[3] Harrison, O. ‘Machine Learning Basics with the K-Nearest Neighbors Algorithm’, towardsdatascience.com, Accessed on 9.04.2022 from https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors- algorithm-6a6e71d01761

[4] Oppermann, A. ‘Optimization Algorithms in Deep Learning’, towardsdatascience.com. Accessed on 8.05.2022 from https://towardsdatascience.com/optimization-algorithms-in-deep-learning-191bfc2737a4

[5] ‘Euclidian Distance Formula’, cuemath.com, Accessed on 9.04.2022 from https://www.cuemath.com/euclidean-distance-formula/

[6] Yiu, T. ‘Understanding Random Forest’, towardsdatascience.com, Accessed on 9.04.2022 from https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors- algorithm-6a6e71d01761

[7] ‘Random Forests From Scratch’, carbonate.github.io, Accessed on 08.05.2022 from https://carbonati.github.io/posts/random-forests-from-scratch/#:~:text=Random%20forests%20are%20essentially%20a,us%20with%20a%20powerful%20classifier