#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Movie Dataset 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# The aim of this project is to run this dataset to produce meaningful and accurate inferences about movies released between 1960 and 2015. Each row contains information about a speific movie on which we would explore. 
# 
# In this report, we would answer a few questions:
# 1. Which movies have the highest profit and also lowest profit?
# 2. Which movie has the highest budget?
# 3. Which movie has the highest runtime?
# 4. Which movie has the highest popularity?
# 5. Is the popularity of a movie related to its profits?
# 6. Which year has the highest profits?
# 7. Which year has the highest release of movies?
# 8. Top 10 Production Companies With Higher Number Of Releases?
# 9. Which genres have the highest releases of movies?
# 10. What Kind Of Properties Are Associated With Movies With High Revenue?
# 11. Top 10 directors who directs most Movies?
# 12. How is popularity trending over time?
# 
# >**NB** Profit is a calculated column, which was calculated by subtacting budget from revenue obtained.

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[39]:


# Load the data
df=pd.read_csv('tmdb-movies.csv')
df.head(1)


# In[40]:


# finding the total number of columns and rows in the data
df.shape


# In[41]:


df.info()


# In[42]:


#since the dataset contain null values also 
#count total rows in each column which contain null values
df.isna().sum()


# In[43]:


df.dropna(inplace=True) # drop null values
df.isnull().sum().any()
df.info(); # check the data


# In[44]:


df.describe() # make summaries about the numerical values in the data
# One of the movies was most popular at 32.98 which is an extreme value
# Also the highest runtime is 705 minutes and the lowewst is 0 minutes which means, it was not recorded
#The least vote count was 10 votes whiles the highest was 9767


# ### Data Cleaning 

# In[45]:


#the the given in the dataset is in string format.
#So we need to change the date in datetime format

df['release_date'] = pd.to_datetime(df['release_date'])
df['release_date'].head()


# In[46]:


sum(df.duplicated())# check duplicates


# In[47]:


df.drop_duplicates(inplace=True)
print("After Removing Duplicate Values (Rows,Columns) : ",df.shape)


# In[48]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section
#we can see that 21 columns in the dataset, We can drop them using drop function.
df.drop(['budget_adj','revenue_adj','keywords','overview','imdb_id','homepage','tagline'],axis =1,inplace = True)
print("After Removing columns that would not be used (Rows,Columns) : ",df.shape)


# In[49]:


df.head(1)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 : Which movies have the highest profit and also lowest profit?
# 

# In[50]:


#calculate Profit for each of the movie
#add a new column in the dataframe name 'Profit'
df['profit'] = df['revenue'] - df['budget']

# movie with the maximum profit
df[(df['profit']==max(df['profit']))].iloc[:,4]


# In[51]:


# movie with the minimum profit
df[(df['profit']==min(df['profit']))].iloc[:,4]


# ### Research Question 2: Which movie has the highest budget?

# In[52]:


# movie with the maximum budget
df[(df['budget']==max(df['budget']))].iloc[:,4]


# ### Research Question 3: Which movie have the highest runtime? 

# In[53]:


# movie with the maximum runtime
df[(df['runtime']==max(df['runtime']))].iloc[:,4]


# ### Research Question 4: Which movie have the highest popularity?

# In[54]:


# movie with the maximum popularity
df[(df['popularity']==max(df['popularity']))].iloc[:,4]


# > The movie with the highest profit is "Avatar"\
# > The movie with the lowest profit is "The Warrior's Way"\
# > The movie with the highest budget is "The Warrior's Way"\
# > The movie with the highest runtime is "Band of Brothers"\
# > The movie with the most popularity is "Jurassic World"

# ### Research Question 5 : Is the popularity of a movie related to its profits?

# In[55]:


# popularity correlated to profits

sns.lmplot(x='profit', y='popularity', data=df);
from scipy import stats
stats.pearsonr(df['profit'], df['popularity'])


# > From the data, popularity of a movie is positively correlated to the amount of profits earned by that movie with a coefficient of 0.6214 which is a strong relationship

# ### Research Question 6 : Which year has the highest profits?

# In[56]:


df.groupby('release_year')['profit'].mean().plot(xticks = np.arange(1960,2016,5))

#setup the title and labels of the figure.
plt.title("Year Vs Average Profit",fontsize = 14)
plt.xlabel('Release year',fontsize = 13)
plt.ylabel('Average Profit',fontsize = 13)

#setup the figure size.
sns.set(rc={'figure.figsize':(10,10)})
sns.set_style("whitegrid")


# > It is observed that movies released in 1997 made the highest profit and 1986 made the least profit

# ### Research Question 7: Which year has the highest release of movies?

# In[57]:


#group data according to their release year and count the total number of movies in each year and plot.
df.groupby('release_year').count()['id'].plot(xticks = np.arange(1960,2016,5))

sns.set(rc={'figure.figsize':(10,5)})
plt.title("Year Vs Number Of Movies",fontsize = 14)
plt.xlabel('Release year',fontsize = 10)
plt.ylabel('Number Of Movies',fontsize = 10)

sns.set_style("darkgrid")


# >After seeing the plot and the output we can conclude that year 2011 year has the highest release of movies followed by year 2009 and 2015.

# ### Reasearch Question 8: Top 10 Production Companies With Higher Number Of Releases?

# In[58]:


def count_genre(x):
    #concatenate all the rows of the genres.
    df_plot = df[x].str.cat(sep = '|')
    data_c = pd.Series(df_plot.split('|'))
    #counts each of the genre and return.
    cnt = data_c.value_counts(ascending=False)
    return cnt

pdtn_cpy = count_genre('production_companies')

#plot the barchart.
pdtn_cpy.iloc[:10].plot(kind='bar',figsize=(10,8),fontsize=13)
plt.title("Production Companies Vs Number Of Movies",fontsize=15)
plt.xlabel('Number Of Movies',fontsize=14)
sns.set_style("whitegrid")


# > The universal pictures released the highest number of movies

# ### Research Question 9: Which genres have the highest releases of movies? 

# In[59]:


ttl_genre = count_genre('genres')
#plot a piechart
ttl_genre.iloc[:10].plot(kind= 'pie',figsize = (13,6),fontsize=12,
        colormap='tab20c',autopct='%1.1f%%', shadow=True, startangle=90) #for beauty

plt.title("Genre With Highest Release",fontsize=15)
plt.xlabel('Number Of Movies',fontsize=13)
plt.ylabel("Genres",fontsize= 13)
sns.set_style("whitegrid")


# > Drama movies had the highest releases

# ### Research Question 10: What Kind Of Properties Are Associated With Movies With High Revenue?

# In[70]:


df1 = df.drop(['id','release_year','runtime','vote_average'], axis=1, inplace=False)
corma= df1.corr().round(2)
sns.heatmap(corma, annot=True);
plt.title('Correlation heatmat of popularity, budget, revenue, vote count and profits', fontsize=15)


# > Profit, vote count, revenue, budget, and popularity are moderately and strongly correlated with each other.
# >
# > Most especially is the Profits and Revenue(0.98), with vote count and revenue being next(0.8).

# ### Research Question 11 : Top 10 Directors Who Directs most Movies?

# In[61]:


count_director_movies = count_genre('director')

#plot a barchart
count_director_movies.iloc[:10].plot(kind='bar',figsize=(13,6),fontsize=12)

#create the title and the labels 
plt.title("Director Vs Number Of Movies",fontsize=15)
plt.xticks(rotation=20)
plt.ylabel("Number Of Movies",fontsize= 13)
sns.set_style("whitegrid")


# > Director John Carpenter directed most movies during the period

# ### Research Question 12: How is popularity trending over time?

# In[62]:


#group data according to their release year and find the mean of the popularity  in each year and plot.
df.groupby('release_year')['popularity'].mean().plot(xticks = np.arange(1960,2016,5))

sns.set(rc={'figure.figsize':(10,5)})
plt.title("Year Vs Popularity Of Movies",fontsize = 14)
plt.xlabel('Release year',fontsize = 10)
plt.ylabel('Popularity',fontsize = 10)

sns.set_style("darkgrid")


# > 1972 had the highest popularity of movies followed by 1977, after which there was relatively lower popular movies in the subsequent years with some rising in popularity from 2010 to 2015

# <a id='conclusions'></a>
# ## Conclusions

# >The movie with the highest profit is "Avatar" and lowest profit is "The Warrior's Way"\
# >The movie with the highest budget is "The Warrior's Way", highest runtime is "Band of Brothers" and most popularity is "Jurassic World"\
# > Popularity of a movie has a high linear relationship with profits earned.\
# > Although movies made in 2011 were the most, movies made in 1977 gained more profits on the average.\
# > Dramas were most released and the Universal Pictures relesed the most movies. Also John Carpenter directed most movies during the period.\
# > Vote count is also highly correlated to revenues earned.\
# > Although recent movies are gaining popularity, movies made in 1972 had massive popularity

# ### Limitations
# 
# > The data contained many missing values which were mostly qualitative so it was difficult trying to fill them, hence I had to drop them. Though they might have affected my analysis and conclusions in a signficant way
