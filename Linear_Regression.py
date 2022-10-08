# part 1-This is done
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.svm._libsvm import predict_proba

diabetes = pd.read_csv('diabetes.csv')
# imports data set and prints the main data, as well as the description of dataset


# print(diabetes)
# print(diabetes.describe())

# part 2
corrMatrix = diabetes.corr()
# print(corrMatrix)

# sb.heatmap(corrMatrix, annot=True)
# plt.show()

# part 3A - countplot
# sb.countplot(x='Outcome', hue='Outcome', data=diabetes)
# plt.legend(labels = ['Negative', 'Positive'])
# plt.tight_layout()
# plt.show()

# part 3B - distribution plot
# sb.displot(diabetes, kind="kde")
# plt.show()

# part 3C - box plot
# sb.boxplot(diabetes[['Glucose', 'SkinThickness', 'BMI', 'Age']])
# plt.show()

# part 3D - Pairplot
# sb.pairplot(diabetes)
# plt.show()

# part 3E - Violin Plot
# sb.violinplot(diabetes[['Outcome']])
# plt.show()

# Part 4 -

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = diabetes['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
model = LogisticRegression()
model.fit(x_train, y_train)
# print(model.score(x_train, y_train))
confMatrix = confusion_matrix(y_train, model.predict(x_train))
# print(confMatrix)
# sb.heatmap(confMatrix / np.sum(confMatrix), annot=True, fmt='.2%', )
# plt.show()

# Part 4 AUC /ROC Curve
# define metrics
# y_pred_proba = model.predict_proba(x_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plt.plot(fpr, tpr, label="AUC=" + str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()

# Part 5 - L1 Regularization, print ROC/AUC Curve + accuracy score, confusion matrix , heatmap
from statistics import mean
from sklearn.linear_model import Ridge, Lasso

cross_val_scores_ridge = []

# List to maintain the different values of alpha
alpha = []

# Loop to compute the different values of cross-validation scores
for i in range(1, 9):
    ridgeModel = Ridge(alpha=i * 0.25)
    ridgeModel.fit(x_train, y_train)
    scores = cross_val_score(ridgeModel, x, y, cv=10)
    avg_cross_val_score = mean(scores) * 100
    cross_val_scores_ridge.append(avg_cross_val_score)
    alpha.append(i * 0.25)
    # Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i]) + ' : ' + str(cross_val_scores_ridge[i]))
ridgeModelChosen = Ridge(alpha=2)
ridgeModelChosen.fit(x_train, y_train)
# confMatrix = confusion_matrix(y_train, model.predict(x_train))


# Evaluating the Ridge Regression model
# print(ridgeModelChosen.score(x_test, y_test))

# ROC/AUC
# y_pred_proba = model.predict_proba(x_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)

# plt.plot(fpr, tpr, label="AUC=" + str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()
# sb.heatmap(ridgeModelChosen / np.sum(cross_val_scores_ridge), annot=True, fmt='.2%', )# plt.show()

# Part 6 - L2 Regularization, print ROC/AUC Curve + accuracy score, confusion matrix , heatmap
# List to maintain the cross-validation scores

cross_val_scores_lasso = []

# List to maintain the different values of Lambda
Lambda = []

# Loop to compute the cross-validation scores
for i in range(1, 9):
    lassoModel = Lasso(alpha=i * 0.25, tol=0.0925)
    lassoModel.fit(x_train, y_train)
    scores = cross_val_score(lassoModel, x, y, cv=10)
    avg_cross_val_score = mean(scores) * 100
    cross_val_scores_lasso.append(avg_cross_val_score)
    Lambda.append(i * 0.25)

# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i]) + ' : ' + str(cross_val_scores_lasso[i]))
lassoModelChosen = Lasso(alpha=.25, tol=0.0925)
lassoModelChosen.fit(x_train, y_train)
y_pred_proba = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
confMatrix = confusion_matrix(y_train, model.predict(x_train))


# plt.plot(fpr, tpr, label="AUC=" + str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()


# Evaluating the Lasso Regression model
# print(lassoModelChosen.score(x_test, y_test))
