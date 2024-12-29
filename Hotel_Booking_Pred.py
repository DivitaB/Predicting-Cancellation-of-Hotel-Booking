#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mp


# In[2]:


get_ipython().system('pip install -U scikit-learn')


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_csv(r'C:/Users/DSB/Desktop/Data_Science_Proj/hotel_bookings.csv') #r means Raw string


# In[5]:


type(df)


# In[6]:


df.head(6)


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(['agent','company'], axis=1, inplace=True)


# In[9]:


df['country'].value_counts() #PRT has highest count, so we take it as mode and replace it w missing values


# In[10]:


df['country'].value_counts().index[0] 


# In[11]:


df['children'].value_counts()


# In[ ]:





# In[12]:


#Imputation
df['country'].fillna(df['country'].value_counts().index[0],inplace=True)


# In[13]:


df['country'].isnull().sum()


# In[ ]:





# In[14]:


df.fillna(0,inplace=True)
df['country']


# In[15]:


df.isnull().sum()


# In[16]:


df[df['children']==0] #passing this filter df[children] in a dataframe df
#this code shows cols where children = 0


# In[17]:


#create a condition
filter1=(df['children']==0) & (df['adults']==0) & (df['babies']==0)


# In[18]:


df[filter1] #shws rows w 0 child, adults, babies


# In[19]:


data=df[~filter1] #data is the var storing cleaned new data


# In[20]:


data


# In[21]:


data.shape #cleaned data with emitted, unwanted rows


# In[22]:


df.shape #old data w errors and everything 180 unwanted rows


# In[23]:


data['is_canceled'].unique() #1 means cancelled and 0 is not


# In[24]:


data[data['is_canceled']==0]['country'].value_counts()


# In[25]:


#to see the % customers who did booking
data[data['is_canceled']==0]['country'].value_counts()/75011*100


# In[26]:


len(data[data['is_canceled']==0])


# In[27]:


country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index() #store this into a DF object
country_wise_data.columns=['country','no.of customers']
country_wise_data
#we get 165 countries


# In[28]:


get_ipython().system('pip install plotly')


# In[29]:


get_ipython().system('pip install chart_studio')


# In[30]:


import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) ##visualise plot in offline mode/notebook mode


# In[31]:


import plotly.express as px


# In[32]:


map_customers=px.choropleth(country_wise_data,locations=country_wise_data['country'],color=country_wise_data['no.of customers'],
              hover_name=country_wise_data['country'],title='Home Country of Customers')


# In[33]:


map_customers.show()


# In[34]:


##Analysing hotel prices across year


# In[35]:


data2=data[data['is_canceled']==0]


# In[36]:


data2


# In[37]:


data2.columns


# In[38]:


#Prob: How much do guests pay for a room per night
sb.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
#Interpretation:
#Prices vary as per the room type, so create a boxplot Room type & adr
#Hue - showing distri of rooms for diff categories of hotel 


# In[39]:


##To determine the most busy months


# In[40]:


data['hotel'].unique()


# In[41]:


data_resort=data[(data['hotel']=='Resort Hotel') & (data['is_canceled']==0)]
data_city=data[(data['hotel']=='City Hotel') & (data['is_canceled']==0)]


# In[42]:


rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['arrival_month','no.of customers visiting']
rush_city


# In[43]:


#for resort i want to see the most no.of busy months, count of people visiting
rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['arrival_month','no.of customers visiting']
rush_resort


# In[44]:


#merge both dataframes
rush_final=rush_city.merge(rush_resort,on='arrival_month')
rush_final.columns=['arrival_month','no.of customers visiting in city', 'no.of customers visiting in resort']
rush_final


# In[45]:


#above, months are not sorted, we ll sort the months


# In[46]:


get_ipython().system('pip install sorted_months_weekdays')
get_ipython().system('pip install sort_dataframeby_monthorweek')


# In[47]:


import sort_dataframeby_monthorweek as sd


# In[48]:


rush_final=sd.Sort_Dataframeby_Month(rush_final,'arrival_month')


# In[49]:


rush_final.columns


# In[50]:


px.line(rush_final,x='arrival_month',y=['no.of customers visiting in city','no.of customers visiting in resort'])


# In[51]:


#above, most busy months are AUG, JULY, SEPT AND MAY


# In[52]:


##AVG Daily rate for cancelled and not cancelled bookings


# In[53]:


data2.columns


# In[54]:


comp=sd.Sort_Dataframeby_Month(data,'arrival_date_month')
sb.barplot(x='arrival_date_month',y='adr',data=comp,hue='is_canceled')
mp.xticks(rotation='vertical')
mp.show()
#orange bar in every month is higher which shows that Avg daily rate is higher for canceled bookings
##higher adr could be the reason for higher booking


# In[55]:


##Analyse bookings made for weekdays/weekends or both


# In[56]:


pd.crosstab(index=data['stays_in_weekend_nights'],columns=data['stays_in_week_nights'])


# In[57]:


def week(row):
    feature1= 'stays_in_weekend_nights'
    feature2='stays_in_week_nights'
    
    if row[feature1]>0 and row[feature2]==0:
        return 'stay_just_weekends'
    elif row[feature1]==0 and row[feature2]>0:
        return 'stay_just_weekdays'
    elif row[feature1]>0 and row[feature2]>0:
        return 'stay_both_weekdays_weekends'
    else:
        return 'unidentified_data'


# In[58]:


data2['weekend_or_weekday']=data2.apply(week,axis=1)


# In[59]:


data2.head()


# In[ ]:





# In[60]:


data2['weekend_or_weekday'].value_counts()

#created our columns


# In[ ]:





# In[61]:


group_data=data2.groupby(['arrival_date_month','weekend_or_weekday']).size().unstack().reset_index()
#unstack() converts into  a table form


# In[ ]:





# In[62]:


group_data


# In[ ]:





# In[63]:


sorted_data=sd.Sort_Dataframeby_Month(group_data,'arrival_date_month')


# In[ ]:





# In[64]:


sorted_data.set_index('arrival_date_month',inplace=True)


# In[ ]:





# In[65]:


sorted_data


# In[ ]:





# In[66]:


sorted_data.plot(kind='bar',stacked=True,figsize=(15,10))


# In[ ]:





# In[67]:


data2.columns


# In[ ]:





# In[68]:


def family(row):
    if (row['adults']>0) & (row['children']>0 or row['babies']>0):
        return 1
    else:
        return 0


# In[ ]:





# In[69]:


data['is_family']=data.apply(family, axis=1) #storing this func in a feature


# In[ ]:





# In[70]:


data.head()


# In[ ]:





# In[71]:


data['total_customers'] = data['adults'] + data['babies'] + data['children']


# In[ ]:





# In[72]:


data.head()


# In[ ]:





# In[73]:


data['total_nights'] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']


# In[ ]:





# In[74]:


data.head(10)


# In[ ]:





# In[75]:


data.columns


# In[ ]:





# In[76]:


data['deposit_type'].unique()


# In[ ]:





# In[77]:


data['deposit_type']


# In[ ]:





# In[78]:


dict={'No Deposit':0, 'Refundable':0, 'Non Refund':1}


# In[ ]:





# In[79]:


data['deposit_given']=data['deposit_type'].map(dict)


# In[ ]:





# In[80]:


data['deposit_given']


# In[ ]:





# In[81]:


data.head()


# In[ ]:





# In[ ]:





# In[82]:


data.columns


# In[ ]:





# In[83]:


data.drop(['adults', 'children', 'babies','deposit_type'],axis=1,inplace=True)


# In[ ]:





# In[84]:


data.columns


# In[ ]:





# In[85]:


data.head()


# In[ ]:





# In[86]:


#Separate categorical cols and num by loop


# In[87]:


cate_features=[col for col in data.columns if data[col].dtype=='object']


# In[ ]:





# In[88]:


data_cat=data[cate_features] #textual data
#we'll perform feature encoding upon data_cat


# In[ ]:





# In[89]:


num_features=[col for col in data.columns if data[col].dtype!='object']


# In[ ]:





# In[90]:


num_features


# In[ ]:





# In[91]:


data[cate_features]


# In[ ]:





# In[92]:


data[num_features]


# In[ ]:





# In[93]:


#group the data on the basis of hotel feature
data.groupby(['hotel'])['is_canceled'].mean().to_dict()
#0.4 & 0.27 val < 0.5 means, more 0s than 1's which means count (1) canceled booking is less than count(0)non-canceled booking


# In[ ]:





# In[94]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:





# In[95]:


data_cat['cancellation']=data['is_canceled'] #--->feature encoding


# In[ ]:





# In[96]:


data_cat.head()
#now we can do feat encoding on top of this dataset


# In[ ]:





# In[97]:


cols=data_cat.columns
#except cancellation feature, we need to encode all other features


# In[ ]:





# In[98]:


cols


# In[ ]:





# In[99]:


cols=cols[0:-1]


# In[ ]:





# In[100]:


cols #now i need to encode all these features stored in 'cols'


# In[ ]:





# In[101]:


#now we iterate on top of this cols list
for col in cols:
    dict2=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict2)


# In[ ]:





# In[102]:


data_cat.head()
#now all the features are converted into numerical data


# In[ ]:





# In[103]:


#merge 2 datas cate_feat and num_feat data
dataframe=pd.concat([data_cat,data[num_features]],axis=1)


# In[ ]:





# In[104]:


dataframe


# In[ ]:





# In[105]:


dataframe.columns #this new dataframe of numerical features


# In[ ]:





# In[ ]:





# In[106]:


#'cancellation', 'is_canceled both are same, so we ll drop any 1
dataframe.drop(['cancellation'],axis=1,inplace=True)


# In[ ]:





# In[107]:


dataframe.columns


# In[ ]:





# In[108]:


#handle outliers
sb.distplot(dataframe['lead_time'])
#right skewness is more, means higher outliers are there


# In[ ]:





# In[109]:


#to reduce outliers, we do Log Transformation; create a function
def handle_outliers(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[ ]:





# In[ ]:





# In[110]:


handle_outliers('lead_time')


# In[ ]:





# In[111]:


#execute a distribution plot
sb.distplot(dataframe['lead_time'])
#now we have less outliers than before


# In[ ]:





# In[112]:


#now we execute Distribution plot for 'adr' feature
sb.distplot(dataframe['adr'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[113]:


dataframe[dataframe['adr']<0] #to see any negative value


# In[ ]:





# In[114]:


##selecting important features using Univariate analysis 
sb.FacetGrid(data,hue='is_canceled',xlim=(0,500)).map(sb.kdeplot,'lead_time',shade=True).add_legend()


# In[ ]:





# In[115]:


##Use co-relation to select imp features
corr=dataframe.corr()


# In[ ]:





# In[116]:


dataframe.columns


# In[ ]:





# In[117]:


corr['is_canceled'].sort_values(ascending=False)


# In[ ]:





# In[118]:


corr['is_canceled'].sort_values(ascending=False).index


# In[119]:


feat_to_drop = ['reservation_status','reservation_status_date','stays_in_weekend_nights',
       'arrival_date_day_of_month','arrival_date_year',
       'arrival_date_week_number']


# In[ ]:





# In[120]:


#High corelation and lowest corealtion features are being deleted
dataframe.drop(feat_to_drop,axis=1,inplace=True) 


# In[ ]:





# In[ ]:





# In[121]:


dataframe


# In[ ]:





# In[ ]:





# In[122]:


dataframe.shape


# In[ ]:





# In[ ]:





# In[123]:


dataframe.isnull().sum()


# In[ ]:





# In[ ]:





# In[124]:


##Feature Selection Approach
#Independent features
x=dataframe.drop('is_canceled',axis=1)


# In[ ]:





# In[ ]:





# In[125]:


#Dep/target feature
y=dataframe['is_canceled']


# In[ ]:





# In[ ]:





# In[126]:


y


# In[ ]:





# In[ ]:





# In[127]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[ ]:





# In[ ]:





# In[128]:


#Lasso(alpha=0.05)


# In[129]:


feat_sel_model=SelectFromModel(Lasso(alpha=0.05))


# In[ ]:





# In[ ]:





# In[130]:


feat_sel_model.fit(x,y)


# In[ ]:





# In[ ]:





# In[131]:


feat_sel_model.get_support() #treat as a filter in col list


# In[ ]:





# In[ ]:





# In[132]:


col=x.columns


# In[ ]:





# In[ ]:





# In[133]:


col


# In[ ]:





# In[ ]:





# In[134]:


selected_features=col[feat_sel_model.get_support()]


# In[ ]:





# In[ ]:





# In[135]:


selected_features


# In[ ]:





# In[ ]:





# In[136]:


x=x[selected_features]


# In[ ]:





# In[ ]:





# In[137]:


y


# In[ ]:





# In[ ]:





# In[138]:


#build ML model


# In[139]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:





# In[140]:


X_train


# In[ ]:


y_train


# In[ ]:


X_test


# In[ ]:


y_test


# In[ ]:


#now we have to predict whether the bookings will get canceled or not on the basis of values in X dataframe


# In[ ]:


X_train.shape


# In[141]:


X_train, X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)


# In[ ]:





# In[ ]:





# In[ ]:





# In[142]:


from sklearn.linear_model import LogisticRegression


# In[143]:


logreg=LogisticRegression() #initialize and stored its object in logreg


# In[ ]:





# In[144]:


#using fit function model could understand some relationship
logreg.fit(X_train,y_train)


# In[ ]:





# In[145]:


#prediction on X testing data
pred=logreg.predict(X_test) #storing in an array


# In[146]:


pred


# In[147]:


#now we have to evaluate how well the ML model is performing
from sklearn.metrics import confusion_matrix #Metrics is a package
confusion_matrix(y_test,pred)


# In[148]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred) #75% accuracy


# In[149]:


from sklearn.model_selection import cross_val_score


# In[151]:


score=cross_val_score(logreg,x,y,cv=10)


# In[152]:


score


# In[153]:


score.mean() #final accuracy is 74% of logreg model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


.set_index('arrival_date_month',inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:




