from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


class Classifier:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logisticregression(self):
        try:
            model = LogisticRegression()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' LogisticRegression ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from Logistic', e)

    def svm(self):
        try:
            model = SVC()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' SVM ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from SVM', e)

    def sgdclassifier(self):
        try:
            model = SGDClassifier()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' SGDClassifier ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from SGDClassifier', e)

    def decisiontree(self):
        try:
            model = DecisionTreeClassifier()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' DecisionTreeClassifier ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from DecisionTree', e)

    def randomforest(self):
        try:
            model = RandomForestClassifier()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' RandomForestClassifier ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from RandomTree', e)

    def adaboost(self):
        try:
            model = AdaBoostClassifier()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' AdaBoostClassifier ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from Adaboost', e)

    def xgboost(self):
        try:
            model = XGBClassifier()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' XGBoostClassifier ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from XGBoost', e)

    def knn(self, n_neighbour):
        try:
            model = KNeighborsClassifier(n_neighbors=n_neighbour)
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' KNeighborsClassifier ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from KNN', e)

    def gaussain_naiv_bayes(self):
        try:
            model = GaussianNB()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' Gaussian Naiv Bayes ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from G_Naiv_Bayes', e)

    def multinomial_naiv_bayes(self):
        try:
            model = MultinomialNB()
            model.fit(self.x_train, self.y_train)
            pred = model.predict(self.x_test)
            conf_mat = confusion_matrix(self.y_test, pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * (recall * precision) / (recall + precision)
            auc = roc_auc_score(self.y_test, pred)
            print("*" * 20 + '==>' + ' Multinomial Naiv Bayes ' + '<==' + '*' * 20)
            print(f'Accuracy score:{model.score(self.x_test, self.y_test)}')
            print(f'F1_Score:{f1_score}')
            print(f'AUC_Score:{auc}')
        except Exception as e:
            print('Exception from M_naiv_Bayes', e)

    def pipeline(self):
        try:
            l_reg = Pipeline(steps=[('reg', LogisticRegression())])
            svm = Pipeline(steps=[('reg', SVC())])
            d_tree = Pipeline(steps=[('reg', DecisionTreeClassifier())])
            r_f = Pipeline(steps=[('reg', RandomForestClassifier())])
            x_g = Pipeline(steps=[('reg', XGBClassifier())])
            nav_g = Pipeline(steps=[('reg', GaussianNB())])
            nav_m = Pipeline(steps=[('reg', MultinomialNB())])
            model = [l_reg, svm, d_tree, r_f, x_g, nav_g, nav_m]
            name = ['LogisticRegression', 'SVM', 'DEcisionTree', 'RandomForest', 'Xgboost', 'Gaussian_Naiv_Bayes',
                    'Multinomial_Naiv_Bayes']
            j=0
            best={}
            best1={}
            for i in model:
                i.fit(self.x_train, self.y_train)
                pred=i.predict(self.x_test)
                auc_score=roc_auc_score(self.y_test,pred)
                score = i.score(self.x_test, self.y_test)
                print(f'{name[j]} score:{score} auc_score:{auc_score}')
                best[name[j]]=score
                best1[name[j]]=auc_score
                j=j+1
            keymax=max(best,key=best.get)
            keymax1=max(best1,key=best1.get)
            print(f'Best Model for this dataset according to accuracy score is:{keymax} ')
            print(f'Best Model for this dataset according to auc_score is:{keymax1} ')
        except Exception as e:
            print('Exception from ', e)



