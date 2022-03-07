#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Importing dataset
#     1.Since data is in form of excel file we have to use pandas read_excel to load the data
#     2.After loading it is important to check null values in a column or a row
#     3.If it is present then following can be done,
#         a.Filling NaN values with mean, median and mode using fillna() method
#         b.If Less missing values, we can drop it as well
# 

# In[2]:


train_data=pd.read_excel('E:\End-2-end Projects\Flight_Price/Data_Train.xlsx')


# In[3]:


train_data.head()


# In[4]:


train_data.info()


# In[5]:


train_data.isnull().sum()


# #### as less missing values,I can directly drop these

# In[6]:


train_data.dropna(inplace=True)


# In[7]:


train_data.isnull().sum()


# In[8]:


train_data.dtypes


# In[ ]:





# #### From description we can see that Date_of_Journey is a object data type,
#      Therefore, we have to convert this datatype into timestamp so as to use this column properly for prediction,bcz our 
#      model will not be able to understand Theses string values,it just understand Time-stamp
#     For this we require pandas to_datetime to convert object data type to datetime dtype.
# 
# 
#     dt.day method will extract only day of that date
#     dt.month method will extract only month of that date

# In[9]:


def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])
    


# In[10]:


train_data.columns


# In[11]:


for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)


# In[12]:


train_data.dtypes


# In[ ]:





# In[ ]:





# In[13]:


train_data['Journey_day']=train_data['Date_of_Journey'].dt.day


# In[14]:


train_data['Journey_month']=train_data['Date_of_Journey'].dt.month


# In[15]:


train_data.head()


# In[ ]:





# In[16]:


## Since we have converted Date_of_Journey column into integers, Now we can drop as it is of no use.
train_data.drop('Date_of_Journey', axis=1, inplace=True)


# In[ ]:





# In[ ]:





# In[17]:


train_data.head()


# In[ ]:





# In[18]:


def extract_hour(df,col):
    df[col+"_hour"]=df[col].dt.hour


# In[19]:


def extract_min(df,col):
    df[col+"_minute"]=df[col].dt.minute


# In[20]:


def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)


# In[ ]:





# In[21]:


# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time
extract_hour(train_data,'Dep_Time')


# In[22]:


# Extracting Minutes
extract_min(train_data,'Dep_Time')


# In[23]:


# Now we can drop Dep_Time as it is of no use
drop_column(train_data,'Dep_Time')


# In[24]:


train_data.head()


# In[ ]:





# In[25]:


# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time

# Extracting Hours
extract_hour(train_data,'Arrival_Time')

# Extracting minutes
extract_min(train_data,'Arrival_Time')

# Now we can drop Arrival_Time as it is of no use
drop_column(train_data,'Arrival_Time')


# In[26]:


train_data.head()


# In[ ]:





# In[27]:


'2h 50m'.split(' ')


# In[ ]:





# #### Lets Apply pre-processing on duration column,Separate Duration hours and minute from duration

# In[28]:


duration=list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:                   # Check if duration contains only hour
            duration[i]=duration[i] + ' 0m'      # Adds 0 minute
        else:
            duration[i]='0h '+ duration[i]       # if duration contains only second, Adds 0 hour
    


# In[29]:


train_data['Duration']=duration


# In[30]:


train_data.head()


# In[31]:


'2h 50m'.split(' ')[1][0:-1]


# In[ ]:





# In[32]:


def hour(x):
    return x.split(' ')[0][0:-1]


# In[33]:


def min(x):
    return x.split(' ')[1][0:-1]


# In[34]:


train_data['Duration_hours']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(min)


# In[35]:


train_data.head()


# In[36]:


train_data.drop('Duration',axis=1,inplace=True)


# In[37]:


train_data.head()


# In[38]:


train_data.dtypes


# In[39]:


train_data['Duration_hours']=train_data['Duration_hours'].astype(int)
train_data['Duration_mins']=train_data['Duration_mins'].astype(int)


# In[40]:


train_data.dtypes


# In[41]:


train_data.head()


# In[42]:


train_data.dtypes


# In[43]:


cat_col=[col for col in train_data.columns if train_data[col].dtype=='O']
cat_col


# In[44]:


cont_col=[col for col in train_data.columns if train_data[col].dtype!='O']
cont_col


# ### Handling Categorical Data

# 
# #### We are using 2 main Encoding Techniques to convert Categorical data into some numerical format
#     Nominal data --> data are not in any order --> OneHotEncoder is used in this case
#     Ordinal data --> data are in order -->       LabelEncoder is used in this case

# In[45]:


categorical=train_data[cat_col]
categorical.head()


# In[46]:


categorical['Airline'].value_counts()


# In[ ]:





# #### Airline vs Price Analysis

# In[47]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Airline',data=train_data.sort_values('Price',ascending=False))


# In[ ]:





# ##### Conclusion-->  From graph we can see that Jet Airways Business have the highest Price., Apart from the first Airline almost all are having similar median

# In[ ]:





# #### Perform Total_Stops vs Price Analysis

# In[48]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Total_Stops',data=train_data.sort_values('Price',ascending=False))


# In[49]:


len(categorical['Airline'].unique())


# In[50]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding
Airline=pd.get_dummies(categorical['Airline'], drop_first=True)
Airline.head()


# In[51]:


categorical['Source'].value_counts()


# In[52]:


# Source vs Price

plt.figure(figsize=(15,5))
sns.catplot(y='Price',x='Source',data=train_data.sort_values('Price',ascending=False),kind='boxen')


# In[53]:


# As Source is Nominal Categorical data we will perform OneHotEncoding


Source=pd.get_dummies(categorical['Source'], drop_first=True)
Source.head()


# In[54]:


categorical['Destination'].value_counts()


# In[55]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination=pd.get_dummies(categorical['Destination'], drop_first=True)
Destination.head()


# In[56]:


categorical['Route']


# In[57]:


categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]


# In[58]:


categorical.head()


# In[59]:


import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[60]:


categorical['Route_1'].fillna('None',inplace=True)
categorical['Route_2'].fillna('None',inplace=True)
categorical['Route_3'].fillna('None',inplace=True)
categorical['Route_4'].fillna('None',inplace=True)
categorical['Route_5'].fillna('None',inplace=True)


# In[61]:


categorical.head()


# In[62]:


#now extract how many categories in each cat_feature
for feature in categorical.columns:
    print('{} has total {} categories \n'.format(feature,len(categorical[feature].value_counts())))


# In[63]:


### as we will see we have lots of features in Route , one hot encoding will not be a better option lets appply Label Encoding


# In[64]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[65]:


categorical.columns


# In[66]:


for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])


# In[67]:


categorical.head()


# In[68]:


# Additional_Info contains almost 80% no_info,so we can drop this column
# we can drop Route as well as we have pre-process that column
    
drop_column(categorical,'Route')
drop_column(categorical,'Additional_Info')


# In[69]:


categorical.head()


# In[70]:


categorical['Total_Stops'].value_counts()


# In[71]:


categorical['Total_Stops'].unique()


# In[72]:


# As this is case of Ordinal Categorical type we perform LabelEncoder
# Here Values are assigned with corresponding key

dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[73]:


categorical['Total_Stops']=categorical['Total_Stops'].map(dict)


# In[74]:


categorical.head()


# In[75]:


train_data[cont_col]


# In[76]:


# Concatenate dataframe --> categorical + Airline + Source + Destination

data_train=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)
data_train.head()


# In[77]:


drop_column(data_train,'Airline')
drop_column(data_train,'Source')
drop_column(data_train,'Destination')


# In[78]:


data_train.head()


# In[ ]:





# In[79]:


pd.set_option('display.max_columns',35)


# In[80]:


data_train.head()


# In[81]:


data_train.columns


# In[ ]:





# ### outlier detection

# In[82]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    


# In[83]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# In[ ]:





# In[ ]:





# #### dealing with Outliers

# In[84]:


data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])


# In[85]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# In[ ]:





# In[86]:


### separate your independent & dependent data


# In[87]:


X=data_train.drop('Price',axis=1)
X.head()


# In[ ]:





# In[88]:


y=data_train['Price']
y


# In[ ]:


##type(X)


# In[ ]:


##type(y)


# In[ ]:


##X.isnull().sum()


# In[ ]:


##y.isnull().sum()


# In[ ]:


#### as now we dont have any missing value in data, we can definitely go ahead with Feature Selection


# In[ ]:





# ### Feature Selection
#     Finding out the best feature which will contribute and have good relation with target variable. 
#     
# ### Why to apply Feature Selection?
#     To select important features to get rid of curse of dimensionality ie..to get rid of duplicate features

# In[ ]:


###np.array(X)


# In[ ]:


##np.array(y)


# ### I wanted to find mutual information scores or matrix to get to know about the relationship between all features.

# ####  Feature Selection using Information Gain,

# In[89]:


from sklearn.feature_selection import mutual_info_classif


# In[ ]:


mutual_info_classif()


# In[ ]:


###mutual_info_classif(np.array(X),np.array(y))


# In[103]:


X.dtypes


# In[90]:


mutual_info_classif(X,y)


# In[81]:


imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
imp


# In[82]:


imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# In[ ]:





# #### split dataset into train & test

# In[116]:


from sklearn.model_selection import train_test_split


# In[117]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:





# In[136]:


from sklearn import metrics
##dump your model using pickle so that we will re-use
import pickle
def predict(ml_model,dump):
    model=ml_model.fit(X_train,y_train)
    print('Training score : {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('predictions are: \n {}'.format(y_prediction))
    print('\n')
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score: {}'.format(r2_score))
    print('MAE:',metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE:',metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    sns.distplot(y_test-y_prediction)
    
    if dump==1:
        ##dump your model using pickle so that we will re-use
        file=open('E:\End-2-end Projects\Flight_Price/model.pkl','wb')
        pickle.dump(model,file)


# In[ ]:





# #### import randomforest class

# In[125]:


from sklearn.ensemble import RandomForestRegressor


# In[137]:


predict(RandomForestRegressor(),1)


# In[ ]:





# In[ ]:





# In[ ]:





# #### play with multiple Algorithms

# In[138]:



from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[ ]:





# In[139]:


predict(DecisionTreeRegressor(),0)


# In[140]:


predict(LinearRegression(),0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Hyperparameter Tuning
#     1.Choose following method for hyperparameter tuning
#         a.RandomizedSearchCV --> Fast way to Hypertune model
#         b.GridSearchCV--> Slow way to hypertune my model
#     
#     2.Assign hyperparameters in form of dictionary
#     3.Fit the model
#     4.Check best paramters and best score

# In[97]:


from sklearn.model_selection import RandomizedSearchCV


# In[98]:


# Number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=6)]

# Number of features to consider at every split
max_features=['auto','sqrt']

# Maximum number of levels in tree
max_depth=[int(x) for x in np.linspace(5,30,num=4)]

# Minimum number of samples required to split a node
min_samples_split=[5,10,15,100]


# In[99]:


# Create the random grid

random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
'max_depth':max_depth,
    'min_samples_split':min_samples_split
}


# In[100]:


random_grid


# In[ ]:


### initialise your estimator
reg_rf=RandomForestRegressor()


# In[101]:


# Random search of parameters, using 3 fold cross validation

rf_random=RandomizedSearchCV(estimator=reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[102]:


rf_random.fit(X_train,y_train)


# In[103]:


rf_random.best_params_


# In[104]:


prediction=rf_random.predict(X_test)


# In[105]:


sns.distplot(y_test-prediction)


# In[106]:


metrics.r2_score(y_test,prediction)


# In[107]:


print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE',metrics.mean_squared_error(y_test,prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[ ]:





# ##### Save the model to reuse it again

# In[ ]:


get_ipython().system('pip install pickle')


# In[108]:


import pickle


# In[ ]:


# open a file, where you want to store the data
file=open('rf_random.pkl','wb')


# In[111]:


# dump information to that file
pickle.dump(rf_random,file)


# In[ ]:





# In[112]:


model=open('rf_random.pkl','rb')
forest=pickle.load(model)


# In[113]:


y_prediction=forest.predict(X_test)


# In[114]:


y_prediction


# In[115]:


metrics.r2_score(y_test,y_prediction)


# In[ ]:





# In[ ]:





# In[ ]:




