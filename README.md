 **All Classifier Algorithm At One Place**<br>
*This is a package that contains all the best classifier algorithms like*
* LogisticRegression
* SVM
* Decision Tree
* Random Forest
* Naiv Bayes
* SGDClassifier
* Xgboost
* Adaboost
* KNN<br>
=>**To access all these classifier you have to implement a simple code rather that importing all these classifier from different package**
<ul>
* Provide the x_train,y_train,x_test,y_test value to the Classifier object
* Now call the classifier function 
* Note: In case of KNN also provide the n_neighbour value
```python
from Classifier import Classifier
clf = Classifier(x_train, y_train, x_test, y_test)
    clf.logisticregression()
    clf.svm()
    clf.sgdclassifier()
    clf.decisiontree()
    clf.adaboost()
    clf.randomforest()
    clf.xgboost()
    clf.knn(3)
    clf.gaussain_naiv_bayes()
    clf.multinomial_naiv_bayes()
```
Output would look like:
```python
********************==> LogisticRegression <==********************
Accuracy score:0.8072916666666666
F1_Score:0.8614232209737829
AUC_Score:0.76680015016894
********************==> SVM <==********************
Accuracy score:0.796875
F1_Score:0.8592057761732852
AUC_Score:0.7328869978726066
********************==> SGDClassifier <==********************
Accuracy score:0.4375
F1_Score:0.30769230769230765
AUC_Score:0.5834063321236391
********************==> DecisionTreeClassifier <==********************
Accuracy score:0.7395833333333334
F1_Score:0.8134328358208955
AUC_Score:0.6865223376298335
********************==> AdaBoostClassifier <==********************
Accuracy score:0.7239583333333334
F1_Score:0.7984790874524715
AUC_Score:0.6794518833687899
********************==> RandomForestClassifier <==********************
Accuracy score:0.7760416666666666
F1_Score:0.8377358490566038
AUC_Score:0.7351395319734702
********************==> XGBoostClassifier <==********************
Accuracy score:0.7395833333333334
F1_Score:0.8076923076923077
AUC_Score:0.7040420473032162
********************==> KNeighborsClassifier <==********************
Accuracy score:0.71875
F1_Score:0.798507462686567
AUC_Score:0.6624953072206232
********************==> Gaussian Naiv Bayes <==********************
Accuracy score:0.765625
F1_Score:0.8351648351648351
AUC_Score:0.7056063070954824
********************==> Multinomial Naiv Bayes <==********************
Accuracy score:0.6510416666666666
F1_Score:0.7545787545787546
AUC_Score:0.5734576398448255
```
=>**If you want's to test you dataset for all classifier algorithm at one call then you have to do this**
```python
from Classifier import Classifier
clf = Classifier(x_train, y_train, x_test, y_test)
clf.pipeline()
```
Output would look like this:
```python
Accuracy score:0.6510416666666666
F1_Score:0.7545787545787546
AUC_Score:0.5734576398448255
LogisticRegression score:0.8072916666666666 auc_score:0.76680015016894
SVM score:0.796875 auc_score:0.7328869978726066
DEcisionTree score:0.734375 auc_score:0.6870854711550494
RandomForest score:0.7708333333333334 auc_score:0.722562883243649
Xgboost score:0.7395833333333334 auc_score:0.7040420473032162
Gaussian_Naiv_Bayes score:0.765625 auc_score:0.7056063070954824
Multinomial_Naiv_Bayes score:0.6510416666666666 auc_score:0.5734576398448255
Best Model for this dataset according to accuracy score is:LogisticRegression 
Best Model for this dataset according to auc_score is:LogisticRegression 
```
By. Manikant Kumar
