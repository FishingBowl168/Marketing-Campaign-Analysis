# Marketing-Campaign-Analysis



Contributors: 
| Contributors    | Github account  | Contirbution    |
|-----------------|-----------------|-----------------|
| Angelina        | Row 1, Col 2    | Row 1, Col 3    |
| Feng Yuan       | Row 2, Col 2    | Row 2, Col 3    |
| Gayathri        | Row 2, Col 2    | Row 2, Col 3    |


## About:

This is a Mini project for SC1015, Titled as Optimizing Marketing Strategies with PCA and Clustering Analysis. Dataset used- Customer personality Analysis (Kaggle). The dataset contained 2240 rows of data both numerical and categorical.The dataset was chosen to analyze customer behavior and identify target groups for promotional campaigns based on various features related to customer demographics, purchasing history, and responses to previous marketing efforts.

## Files Included:

1,Slides:The Powerpoint slides for the video.

2,Jupyter Notebook
  
3,DataSets:Contained within is the dataset sourced from kaggle.





## Problem Definition:

To maximise company profits, how can we succinctly assess individual features' influence on expenditure and promotional outcomes? 




## Data Preparation:

Fill missing values with median 
- Income was the only one with missing values. Due to high outliers, we used median to fill the NULL values. 
Increase interpretability 
-Converted the 'Dt_Customer' column from object to datetime format to calculate the duration of each customer's enrollment with the company, which improves interpretability.
- Explored the unique values in the 'Marital_Status' and 'Education' columns.  
- Conducted abstraction and simplify values to either alone/partner and undergraduate/graduate/Post graduate
- Changed feature names like amtwinebought to wine to improve readability
- Convert year of birth to age

Remove useless information
- Z_CostContact and Z_revenue only has one unique value
- Customer ID is not a feature that would affect expenditure 

Ordinal encoding 
- We find columns with objects as datasets: Education and Living_with
- Conducted ordinal encoding to convert categorical data to numerical data
- Education: Undergraduate=1, graduate=2, postgraduate=3
- Living with: Alone=1, Partner=2

### FOR CLUSTERING:
Principal of Component analysis (PCA)
- improve efficiency by compressing the predictor data using PCA. Since there are 24 features. We use PCA to reduce dimensionality 
- Evaluate explained variance ratio to decide whether we want it to be reduced to 2 or 3 dimensions 

Standard scaling 
- It involves transforming numerical features so that they have a mean of 0 and a standard deviation of 1. 
- Since clustering relies on distance metric, they are sensitive to the scale of features. Standard scaling ensures that distances between data points are calculated based on meaningful comparisons, rather than being dominated by features with larger scales.

## Models Used:
Model 1: Linear Regression suggests family size, income, age, and parenthood are key predictors of customer expenditure.
Model 2: Logistic Regression indicates age and income are significant over other factors.
Model 3: GBM Regressor shows family size and income are more critical than in Model 2, with better expenditure prediction accuracy.
Model 4: Hill Climbing focuses on maximizing profits using family size and income, finding optimal points for profit maximization.
Models 5-7: Clustering (Agglomerative, K-means, DBSCAN) identifies four main target groups and highlights age, income, and family size as proportional to expenditure. Agglomerative Clustering, chosen as the best fit, evenly considers all dataset features, making it a suitable expenditure predictor.
Model 8: Random Forest Classifier identifies features like wine, income, and expenditure as key to impacting accepted promotions.
Model 9: KNN is useful for customer recommendations on websites, enhancing customer engagement.
Agglomerative Clustering could be the most suitable for predicting customer features due to its holistic approach in considering all features equally.


## Conclusion:
How can we succinctly assess individual features' influence on expenditure and promotional outcomes?
Overall, the models point to age, income and family size as suitable features on expenditure. With income being the best indicator throughout all models. While Random Forest classifier points to wine, income and spent to be important features in predicting promotional outcome. 


| Model                                | Accuracy                        | 
|--------------------------------------|---------------------------------|
| Linear Regression                    | 0.6711                          |
| Logistic Regression                  | 0.8259                          | 
| Gradient Boosting Machine(regressor) | 0.7202(X=Income,y=Spent)        |
| Random forest(Classifier)            | 0.9531(Predicted=AcceptedCmp1)  |


| Model :Clustering                    | Silhouette Score          | 
|--------------------------------------|---------------------------|
| Agglomerative clustering             | 0.48373                   |
| Kmeans                               | 0.43050                   | 
| DBSCAN                               | 0.42173                   |


## Outcome and what can the seller do to improve:
1, add the recommendation system (generated from knn) to their website
2, meat and wine bring the most revenue
3,a focus on websites and online promotions may work well in improving revenue for company
4, campaigns did better than deals though it decreased over time. Companies can improve on promotions and make them more appealing to ensure continued participation

## what did we learn from this project?:

The key learnings are understanding the dataset, handling data issues, performing exploratory analysis, and setting up the groundwork for building machine learning models to gain insights into customer purchasing behavior and promotion effectiveness for optimizing a company's revenue. We got to learn new models apart from lesson such as: 
Gradient Boosting Machines 
Logistics
Hill Climbing
Agglomerative
K-Mean
DBSCAN
KNN
Random Forest Classifer








References:
Bonthu, H. (2023, April 18). KModes clustering algorithm for categorical data. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/


scikit-learn developers. (2023). sklearn.preprocessing.LabelEncoder.Retrieved April 22, 2023, from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

Tran, J. (2021, December 14). Random Forest Classifier in Python - Towards Data Science. Medium. https://towardsdatascience.com/my-random-forest-classifier-cheat-sheet-in-python-fedb84f8cf4f


Contribution
1, codes: Angelina, Fengyuan, Gayathri
2, slides and videos:  Angelina, Fengyuan, Gayathri
3, script and readme: Angelina, Fengyuan, Gayathri




