# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
DEVELOPED BY:Haridharshini.S
REG NO:212221230033
```
```
DATA TO TRANSFORM CSV:
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats 
df=pd.read_csv("Data_To_Transform.csv")  
df 
df.skew() 

#Log Transformation  
np.log(df["Highly Positive Skew"])  

#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])

#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])

#Square Transformation  
np.square(df["Highly Negative Skew"])

# POWER TRANSFORMATION
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df 
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df

#QUANTILE TRANSFORMATION:  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show() 
df.skew()  
df 
```
```
TITANIC.CSV
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  

#ReciprocalTransformation  
np.reciprocal(df["Age"])  

#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  


#QUANTILE TRANSFORMATION  

from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  


df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  

sm.qqplot(df['Age_1'],line='45')  
plt.show()  

df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  

sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df  
```
# OUPUT
##Data_to_transform csv
![q1](https://user-images.githubusercontent.com/94168395/169068864-4559a450-ea9b-469d-87e4-3f82da097d79.png)
![q2](https://user-images.githubusercontent.com/94168395/169071389-ae600fc3-724f-4960-8705-c8bfaa74b9c6.png)
![q3](https://user-images.githubusercontent.com/94168395/169071419-a154e5f3-30bb-4682-b819-95fe4803129e.png)
![q4](https://user-images.githubusercontent.com/94168395/169071474-8bbae050-7826-415c-8fcf-15760305364a.png)
![q5](https://user-images.githubusercontent.com/94168395/169072810-6419e89e-cd6c-4b20-a61c-df495fef526f.png)
![q6](https://user-images.githubusercontent.com/94168395/169072836-629ddb28-0cac-48d9-8925-fd795cee6f32.png)
![q7](https://user-images.githubusercontent.com/94168395/169072885-6a6325ab-4a7b-485a-ae45-341b3f645bd2.png)
![q8](https://user-images.githubusercontent.com/94168395/169072917-769dabff-2162-45e3-8dd0-f34a7322774f.png)
![q9](https://user-images.githubusercontent.com/94168395/169072964-d718b0dd-c4e7-45ad-8891-011e9f65a98a.png)
![q10](https://user-images.githubusercontent.com/94168395/169073002-44c827e9-d865-4954-b056-b807a2f79214.png)
![q11](https://user-images.githubusercontent.com/94168395/169073359-bc6e25ca-d733-4c6f-9c9b-6487b23f0708.png)
![q12](https://user-images.githubusercontent.com/94168395/169073488-d44c1daa-8237-4b65-b40c-a5524934547d.png)
![q13](https://user-images.githubusercontent.com/94168395/169073531-a71941f6-1a69-4869-987f-d8582d64a002.png)
![q14](https://user-images.githubusercontent.com/94168395/169073562-29a51e15-4fc2-480a-9919-e29196670244.png)
![q15](https://user-images.githubusercontent.com/94168395/169073617-e1bb02ee-46c6-4f5f-8639-a25e203baa5f.png)
![q16](https://user-images.githubusercontent.com/94168395/169073698-86da43de-5f5f-4311-b636-ef39c3dc54e1.png)
![q17](https://user-images.githubusercontent.com/94168395/169073839-b80ba813-2181-441b-8c01-6e48539a6a7d.png)
![q18](https://user-images.githubusercontent.com/94168395/169073893-781255b3-7c92-4cc5-945b-b9b747eec65c.png)
##titanic_csv
![a1](https://user-images.githubusercontent.com/94168395/169074228-073b3f26-2cc1-49e4-ba10-365f42c5889f.png)
![a2](https://user-images.githubusercontent.com/94168395/169074263-6ec81fac-ef40-4437-b67c-3cbc8006529f.png)
![a3](https://user-images.githubusercontent.com/94168395/169074312-7aa73d66-5499-4420-98d8-dc2cab401588.png)
![a4](https://user-images.githubusercontent.com/94168395/169074355-b08cef95-2a08-48bf-a0e8-6e256f9b692e.png
![a5](https://user-images.githubusercontent.com/94168395/169074539-5c4046d4-856d-41b5-b8b0-be7e17971c90.png)
![a6](https://user-images.githubusercontent.com/94168395/169074581-a8968295-f0f1-4137-85a4-e3f9e4baf567.png)
![a7](https://user-images.githubusercontent.com/94168395/169074605-61b47838-ae90-463a-8055-91d2397e2f93.png)
![a8](https://user-images.githubusercontent.com/94168395/169074640-8a032819-26ae-4a47-a5e9-8d550827e6ae.png)
![a9](https://user-images.githubusercontent.com/94168395/169074681-e9420fcf-a5f4-42c6-befd-dcda3889c2c4.png)
![a10](https://user-images.githubusercontent.com/94168395/169074712-b122dd30-0274-4523-8312-6da4ca8b3ba9.png)
![a11](https://user-images.githubusercontent.com/94168395/169074734-65933e2f-aefc-4ed7-84e2-6d47a9a07c39.png)
![a12](https://user-images.githubusercontent.com/94168395/169074773-66db75e0-801c-44da-b08e-fde5b30664cf.png)
![a13](https://user-images.githubusercontent.com/94168395/169074793-83726d3f-3651-4b01-8abb-7b110a4ad45a.pn
![a14](https://user-images.githubusercontent.com/94168395/169075187-c4768a4a-849d-4d7f-86f2-ac664525b862.png)
![a15](https://user-images.githubusercontent.com/94168395/169075220-7ef48e20-708d-4f3d-9329-63556b544e2f.png)

#OUTPUT:
therefore Feature transformation techniques is been performed on given dataset and saved into a file successfully.

