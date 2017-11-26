## Introduction

Since I don't have a lot of ideas let's see what other people have done for this dataset. The following are key points form other peoples kernels.

---

### [Megan Risdal](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)

*Feature engineering*
 - Break families into 3 groups (plot family size vs survival - barplot)
 - Separate the passangers with respect to their decs (from the Cabin variable)
 - Child and mother bins (plot age histogram + survival)

*Implanting missing data*
 - Given the small data set do not delete
 - Implant missing Embarked data based on passenger class and fare (replace the NA values with 'C')
 - Implant missing Fare value (maybe use median)
 - Implant age using recursive partitioning for regression model (Look up mice implantation)
 
*Model*
 - Random forrest
 - Show model error (plot)
 - Plot variable importance
 
*Notes*
 - Nice format (structure) of the kernel
 - She has index, which is nice


### [swamysm](https://www.kaggle.com/swamysm/beginners-titanic)

 - Interesting conclusion - 
 ```When I submit the predicted survival data from various models that built in the course to Kaggle competion, i have got approximately the same score. Now I realize that why data scientist used to spend most of their time into feature engineering and exploratory analysis compare to actual model building. Model that we are using is definitely important, however more than that understanding our data and feature engineering is crucial.```


### [Anisotropic](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

```
Method of ensembling (combining) base learning models, in particular the variant of ensembling known as Stacking
```

*Feature engineering*
 - Feature that tells whether a passenger had a cabin on the Titanic
 - FamilySize as a combination of SibSp and Parch
 - Feature IsAlone from FamilySize
 - Random Age based on mean and std

*Visualization*
 - Pearson Correlation Heatmap
 - Takeaway from the Plots

*Model (Ensembling & Stacking)*
 - RandomForestClassifier
 - AdaBoostClassifier
 - GradientBoostingClassifier
 - ExtraTreesClassifier
 - SVC
 - KFold


### General notes
 - A lot of people use RandomForrest even though it is known to overfit
 
 ---
 
## Goals
 - Construct pipeline that does the preprocessing and learning
 - Get higher than `2793/8677` in kaggle
 
 ---
 
## Plan
 - Apply regression to fill the missing age values
 - Input the median value for the orther missing feature values
 - One-hot encode Gender and Embarked
 - Apply binning to age
 - Visualize different features against survival rate
 - Visualize confusion matrix
 - Construct pipeline for feature mapping
 - Try RandomForrest, SVM and Logistic regression
 - Vizualize model error and variable importance
 - Grid search for the best hyper-parameters
 - Apply model on test data and submit
 - Profit
