#!/usr/bin/env python
# coding: utf-8

# ### Problem #1

# In[14]:


import pickle
from pandas_profiling import ProfileReport
import bz2
import random
import os
import pandas as pd


# In[5]:


ifile = bz2.BZ2File("R3K_Daily.bz2",'rb')
week_1_df = pickle.load(ifile)
ifile.close()


# In[6]:


p = ProfileReport(week_1_df)
p.to_file("ProfileReportAllFields.html")


# In[7]:


week_1_df.head()


# ##### The Pickle format is more efficient than CSV formatting. It is faster and lighter than CSVs and allows the user to serialize any type of python object (not solely data). I think the main advantage as it concerns this course is that you can use the pickle functionality to save machine learning models.
# 
# ##### One advantage dataframes have over numpy arrays is that arrays can only contain data objects of the same types, while dataframes can contain multiple datatypes. For machine learning models this presents obvious advantages as we can use the dataframe object to evaluate numerical data as well as text data. 

# ### Problem 2

# In[8]:


dual_class_stocks = []
symbol_list = week_1_df['Symbol'].unique().tolist()
for sym in symbol_list:
    if "." in sym:
        dual_class_stocks.append(sym)


# In[9]:


print(dual_class_stocks)


# ##### An example of Dual class stocks contained in the data would be BF.A and BF.B which are different classes of stock in the Brown-Forman Corp

# ### Problem 3

# In[10]:


minimum_close = week_1_df[week_1_df['Close'] == week_1_df['Close'].min()]


# In[11]:


minimum_close


# ##### The minimum closing price was .008893 for the stock WETF on October 25, 2004. Each exchange has there own rules about minimum values required to retain listing eligibilty. It is likely this stock would be traded over the counter or on the pink sheets, but I couldnt find any SEC rules online prohibbiting stocks from trading for less than a penny. The cause of such low prices would be that equity in the company is not very valuable.

# ### Problem 4

# ##### What does “Rejected” mean next to the line “High is highly correlated with Close”? This isn't present in the profiling report, but it seems pretty self explanatory that any column that displayed that shows a low correlation coefficient with the 'Close' Column.
# 

# ### Problem 5

# In[12]:


kors_df = week_1_df[week_1_df['Symbol'].isin(['KORS'])].rename(columns={"Close": "Kors_Close"})
capri_df = week_1_df[week_1_df['Symbol'].isin(['CPRI'])].rename(columns={"Close": "Capri_Close"})


# In[15]:


fashion_merged_df = pd.merge(kors_df[['Date','Kors_Close']],capri_df[['Date','Capri_Close']],on='Date', how='outer')


# In[16]:


fashion_merged_df.head()


# In[17]:


fashion_merged_df.tail()


# In[18]:


fashion_merged_df.plot.line(x='Date', y=['Kors_Close','Capri_Close'], subplots=True)


# In[19]:


fashion_merged_df.plot.line(x='Date', y=['Kors_Close','Capri_Close'])


# In[20]:


print(round(fashion_merged_df['Kors_Close'].corr(fashion_merged_df['Capri_Close']),4))


# ##### Yes it appears that the person who compiled this dataset backfilled all the data concerning 'CPRI''s value back to 2012 (to match that of 'KORS'  despite the fact that it did not exist until 2018. I think the proper way to handle this would be to treat them as distinct entities and leave all the values prior to 2018 for CPRI asa NaNs and everything post 2018 for 'KORS' as NaNs. 

# ### Problem 6

# In[21]:


goog_df = week_1_df[week_1_df['Symbol'].isin(['GOOG'])].rename(columns={"Close": "GOOG_Close"})
googl_df = week_1_df[week_1_df['Symbol'].isin(['GOOGL'])].rename(columns={"Close": "GOOGL_Close"})


# In[22]:


search_merged_df = pd.merge(goog_df[['Date','GOOG_Close']],googl_df[['Date','GOOGL_Close']],on='Date', how='outer')


# In[23]:


search_merged_df.head()


# In[24]:


search_merged_df.tail()


# In[25]:


search_merged_df.plot.line(x='Date', y=['GOOG_Close','GOOGL_Close'], subplots=True)


# In[26]:


search_merged_df.plot.line(x='Date', y=['GOOG_Close','GOOGL_Close'])


# In[27]:


print(round(search_merged_df['GOOG_Close'].corr(search_merged_df['GOOGL_Close']),4))


# ##### While Goog and GOOGL nearly identical, from my understanding GOOGL granted its holders voting rights in the entity while GOOG did not, making the former very slightly more valuable. I think the way they are trated here as seperate entities is the proper way to handle this data. In fact due to the fact that they are both equity stakes in the same underlying entity, it is important to view them seperately as they are here, so that if they deviate from each other for some sort of supply/demand driven reason you could capitize on a reversion. 

# ### Problem 7

# In[36]:


gov_df = week_1_df[week_1_df['Symbol'].isin(['GOV'])].rename(columns={"Close": "GOV_Close"})
opi_df = week_1_df[week_1_df['Symbol'].isin(['OPI'])].rename(columns={"Close": "OPI_Close"})


# In[37]:


real_estate_merged_df = pd.merge(gov_df[['Date','GOV_Close']],opi_df[['Date','OPI_Close']],on='Date', how='outer')


# In[40]:


real_estate_merged_df.info()


# In[46]:


real_estate_merged_df.tail(10)


# In[48]:


print(round(real_estate_merged_df['GOV_Close'].corr(real_estate_merged_df['OPI_Close']),4))


# In[50]:


real_estate_merged_df.plot.line(x='Date', y=['GOV_Close','OPI_Close'])


# ###### This one is a little more difficult for me to see what the correct way to evaluate this is. It appears the person that compiled the data backfilled all data for OPI based off the reverse stock split by multiplying the value of GOV stock by 4 and making some slight adjustment that I can not understand (maybe due to dividends or doing some type of discounting the value off the risk free rate to the dates prior to the OPI entities existence?). Similar to the KORS/CAPRI example I think the proper way to treat this would be as two seperate entities, since OPI presumably has a different risk reward post merger than GOV did pre merger after adding an entirely new company into the fold.

# ### Problem 8

# ###### When there are multiple calsses of a stock I think generally it would make sense to include them both in a model. I understand that having 2 fields in a model that are perfectly correlated may cause colinearity problems in certain models, but I think most real ML models are more complex than the simple linear regression types where colinearity would cause issues. Also I think adding multiple classes of stock may add allow for the model to find opportunities it otherwise wouldnt in some cases. For instance, Berkshire Hathway's A shares cost somewhere around 400,000 each and their B shares cost ~400 dollars each. If available is being used as a constraint in the model, this would allow more flexibility in suggesting a position in Berkshire Hathaway. Also, although the shares represent equity in the same company, they still have different shareholders subject to different liquidity risks, etc. and it is conceivable that the 2 shares could deviate enough due to a supply/demand imbalance that there is an arbitrage opportunity between the two classes. 

# ### Problem 9

# ###### Survivorship bias is the bias in a dataset when it only considers 'surviving' stocks in a portfolio's performance (eg excluding stocks like Lehman Brothers, Enron, Woldcom, etc. from the dataset). It will alter the portfolio's performance in a way incinsistent with reality. A good example was in this weeks lecture, where a porftfolio was constructed solely of pinksheets stocks, which can be extraordinarily volatile. The ten stocks included in the lecture example had a large negative return over the time frame, but when you excluded all of the equities that went to zero and replaced them with ones that had 'survvied', it showed a large postive return, despite the fact that you obviously wouldn't know which ones were going to go to zero when the portfolio was constructed. To test for survivorship bias in the dataset, I would first look at what portion of the dataset's last clsoing price was not the date of the last record in the dataset (it looks like in this case that would be 127 of the 1056 total symbols), then see if any of those symbols lost all of their value while during the timeframe this dataset accounts for. It looks like there are 13 stocks that had a last close date earlier than the last recorded date in the data set that lost 99%+ of their value, so I think dataset does not have survivorship bias (see work below).

# In[106]:


week_1_df.head()


# In[107]:


mins_and_max_df = week_1_df.groupby('Symbol').Close.agg(['min', 'max']) 


# In[111]:


first_record_df = week_1_df.groupby('Symbol').nth(0).reset_index()
first_record_df = first_record_df[['Symbol','Date', 'Close']].rename(columns={"Date":"First_Date","Close":"First_Close"})
last_record_df = week_1_df.groupby('Symbol').nth(-1).reset_index()
last_record_df = last_record_df[['Symbol','Date', 'Close']].rename(columns={"Date":"Last_Date","Close":"Last_Close"})
#first_record_df.head()


# In[112]:


first_and_last_df = pd.merge(first_record_df,last_record_df,on='Symbol', how='outer')
first_and_last_df = first_and_last_df[['Symbol', 'First_Date', 'Last_Date', 'First_Close', 'Last_Close']]


# In[113]:


first_and_last_df['Perc_Change'] = round(((first_and_last_df['Last_Close'] - first_and_last_df['First_Close'])/first_and_last_df['First_Close'])*100,2)


# In[114]:


first_and_last_df.info()


# In[116]:


not_last_date_available_df = first_and_last_df[first_and_last_df['Last_Date'] != '2019-03-08']


# In[117]:


not_last_date_available_df.info()


# In[120]:


not_last_date_available_df.sort_values(by = ['Perc_Change']).head(20)


# In[ ]:




