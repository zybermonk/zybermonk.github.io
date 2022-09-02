---
layout: page
title: Housing price prediction
description: an inter university kaggle competetion
img: assets/img/projects/kaggle/housing_crash.jpg
importance: 1
category: fun
---
- Author: Mohan Ramesh
- University of Limerick private competetion
- P.S: Git repo will be linked here soon

# Summary:

House pricing is almost as much an art as a theory. There are many contributing factors that decide the price of the house and many approaches to solve the difficulty of the issue. In this research, I studied the housing market from a numerical and non-numeric standpoint and was able to formulate a range of approaches that were aimed at price forecasts.
What I found most surprising is the propensity of homebuyers to give preference to the visual aspects and their feeling about the area when making their buying decisions. Many psychological experiments have been conducted about what helps people decide to buy the single most significant purchase of their lives, but only a few studies have used graphics in their price forecasts. Since half the decision has been taken by our sense of sight, I decided to do housing image classification and house pricing by using numerical data,categorical data, text description(advertising) along with its corresponding image.

Below is a Schematic I created to represent the model we see throughout this notebook.

<div>
{% include figure.html path="assets/img/projects/kaggle/schema.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

### XGBoost stands for extreme Gradient Boosting.

The name xgboost, though, actually refers to the engineering goal to push the limit of computations resources for boosted tree algorithms. Which is the reason I make use of xgboost to solve our problem.

The house price dataset we are using includes Numerical data, Categorical data, Text data and Image data as well.
So, we tackle each type of data differently. 
- The first 3 types of data are tackled locally using Feature Engieering methods, Different types of encoding, NML decomposers, Randomized search for selecting best hyperparameters. 
- Whereas the image data is trained using Google Colab's TPU by a pre-trained Deep Neural Network called ResNet-50, and features are extracted at the pre-final layer ready to be fed to the XGBoost regressor.

So, This Solution follows a Basic Data Visualization, followed by Feature Engineering, Then Building a model to train and finally Deploying it. Let's look at every step, one by one:

# Data Visualization


```python
'''Import necessary libraries '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

see what data looks like. First few rows.


```python
df=pd.read_csv('train.csv')
df.head()
```



<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ad_id</th>
      <th>area</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>ber_classification</th>
      <th>county</th>
      <th>description_block</th>
      <th>environment</th>
      <th>facility</th>
      <th>features</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>no_of_units</th>
      <th>price</th>
      <th>property_category</th>
      <th>property_type</th>
      <th>surface</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>996887</td>
      <td>Portmarnock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dublin</td>
      <td>A SELECTION OF 4 AND 5 BEDROOM FAMILY HOMES LO...</td>
      <td>prod</td>
      <td>NaN</td>
      <td>None</td>
      <td>53.418216</td>
      <td>-6.149329</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>new_development_parent</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>999327</td>
      <td>Lucan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dublin</td>
      <td>**Last 2 remaining houses for sale ***\n\nOn v...</td>
      <td>prod</td>
      <td>NaN</td>
      <td>None</td>
      <td>53.364917</td>
      <td>-6.454935</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>new_development_parent</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>999559</td>
      <td>Rathfarnham</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dublin</td>
      <td>Final 4 &amp;amp; 5 Bedroom Homes for Sale\n\nOn V...</td>
      <td>prod</td>
      <td>NaN</td>
      <td>None</td>
      <td>53.273447</td>
      <td>-6.313821</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>new_development_parent</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9102986</td>
      <td>Balbriggan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dublin</td>
      <td>Glenveagh Taylor Hill, Balbriggan\n\r\n*Ideal ...</td>
      <td>prod</td>
      <td>NaN</td>
      <td>None</td>
      <td>53.608167</td>
      <td>-6.210914</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>new_development_parent</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9106028</td>
      <td>Foxrock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dublin</td>
      <td>*New phase launching this weekend Sat &amp;amp; Su...</td>
      <td>prod</td>
      <td>NaN</td>
      <td>None</td>
      <td>53.262531</td>
      <td>-6.181527</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>new_development_parent</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Check for NULL values


```python
df.isnull().sum()
```




    ad_id                    0
    area                     0
    bathrooms               51
    beds                    51
    ber_classification     677
    county                   0
    description_block        0
    environment              0
    facility              2017
    features                 0
    latitude                 0
    longitude                0
    no_of_units           2923
    price                   90
    property_category        0
    property_type           51
    surface                551
    dtype: int64



We can see a few columns have a lot of NULL values. We need to Check if that's the case for test data set as well.
Then we have to Handle these NULL values accordingly.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2982 entries, 0 to 2981
    Data columns (total 17 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   ad_id               2982 non-null   int64  
     1   area                2982 non-null   object 
     2   bathrooms           2931 non-null   float64
     3   beds                2931 non-null   float64
     4   ber_classification  2305 non-null   object 
     5   county              2982 non-null   object 
     6   description_block   2982 non-null   object 
     7   environment         2982 non-null   object 
     8   facility            965 non-null    object 
     9   features            2982 non-null   object 
     10  latitude            2982 non-null   float64
     11  longitude           2982 non-null   float64
     12  no_of_units         59 non-null     float64
     13  price               2892 non-null   float64
     14  property_category   2982 non-null   object 
     15  property_type       2931 non-null   object 
     16  surface             2431 non-null   float64
    dtypes: float64(7), int64(1), object(9)
    memory usage: 396.2+ KB
    

Here in the above result we see that the dataset is a mixed type of dataset, and but manually looking at the columns we see that bathrooms and beds are actually categories, but is given as a number. The columns county and environment are actually useless as they contain 1 unique category only. And the column no_of_units is not really necessary because almost all of the values are NULL, which we later see that no_of_units in the test dataset is actually empty. Then we can also decide to keep the text features like description_block, features and facility aside for a while, and only look at the rest of the features.


```python
numerical = ['price', 'latitude', 'longitude', 'surface']
categorical = ['area', 'ber_classification', 'property_category', 'property_type']
```


```python
df[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4));
```

<div>
{% include figure.html path="assets/img/projects/kaggle/output_16_0.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

We see all of the Numerical values are skewed


```python
fig, ax = plt.subplots(2, 2, figsize=(15, 6))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
```

<div>
{% include figure.html path="assets/img/projects/kaggle/output_18_0.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

Now we have a general idea of what the data looks like and lets proceed to Feature Engieering and Feature selection.

# Feature Engineering and Selection

We already discarded several columns by manual evaluation of train and test sets, and we also know what are categorical features and what are Numerical. So lets handle these columns individually.

The 'ad_id' column is to be dropped logically since it is an unique value. But I decide to keep it in this case, as I noticed a strange relationship between the ad_id and the price. As there is a common prefix among group of houses.

bathrooms and beds columns are actually categories so I change their dtype


```python
df['bathrooms'] = df.beds.astype(object)
df['beds'] = df.beds.astype(object)
```

Now I fill all the NULL values:
- Categorical : With Mode()
- Numerical   : With Mean()
of the respective columns.

And drop the redundant columns, inferred manually.


```python
df['bathrooms']=df['bathrooms'].fillna(df['bathrooms'].mode()[0])
df['beds']=df['beds'].fillna(df['beds'].mode()[0])
df['ber_classification']=df['ber_classification'].fillna(df['ber_classification'].mode()[0])
df.drop(['county'],axis=1,inplace=True)
df.drop(['environment'],axis=1,inplace=True)
df["facility"].fillna("low", inplace = True)
df.drop(['no_of_units'],axis=1,inplace=True)
df['price']=df['price'].fillna(df['price'].mean())
df['property_type']=df['property_type'].fillna(df['property_type'].mode()[0])
df['surface']=df['surface'].fillna(df['surface'].mean())
```

lets plot the features to see if there are any null values


```python
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f1e0b147c0>




<div>
{% include figure.html path="assets/img/projects/kaggle/output_28_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>


### Now lets load the test dataset, which has undergone the same features engineering methods as train data


```python
test_df = pd.read_csv('test_prep.csv') # test data is handled seperately and only shown here.
```


```python
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='summer')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f1e1449280>




<div>
{% include figure.html path="assets/img/projects/kaggle/output_31_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>


### Now Lets start encoding

Before encoding we make sure encoding is being performed on both train and test sets, because there might be categories which are different (more/less) from each other.


```python
final_df=pd.concat([df,test_df],axis=0)
```

# Label Encoding the Ordinal Categories


```python
final_df["bathrooms"] = final_df["bathrooms"].astype('category')
final_df["beds"] = final_df["beds"].astype('category')
final_df["ber_classification"] = final_df["ber_classification"].astype('category')
final_df["facility"] = final_df["facility"].astype('category')
final_df["bathrooms"] = final_df["bathrooms"].cat.codes
final_df["beds"] = final_df["beds"].cat.codes
final_df["ber_classification"] = final_df["ber_classification"].cat.codes
final_df["facility"] = final_df["facility"].cat.codes
```

# One Hot Encoding the Nominal Categories


```python
columns=['area','property_category','property_type']

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

final_df=category_onehot_multcols(columns)
```

    area
    property_category
    property_type
    


```python
final_df.shape
```




    (3482, 177)



### Now We include Text features.


```python
nmf_df = final_df
```

we vectorize each of the text columns and tokenize the top words in them


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf1 = TfidfVectorizer(max_df=0.90, min_df= 4, stop_words="english")
dtm_with_tfidf_des = tfidf1.fit_transform(nmf_df['description_block'])
dtm_with_tfidf_des
```




    <3482x6207 sparse matrix of type '<class 'numpy.float64'>'
    	with 544865 stored elements in Compressed Sparse Row format>




```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer(max_df=0.90, min_df= 4, stop_words="english")
dtm_with_tfidf_fea = tfidf2.fit_transform(nmf_df['features'])
dtm_with_tfidf_fea
```




    <3482x1441 sparse matrix of type '<class 'numpy.float64'>'
    	with 62245 stored elements in Compressed Sparse Row format>




```python
from sklearn.decomposition import NMF
nmf_model1 = NMF(n_components=10, random_state=1)
nmf_model1.fit(dtm_with_tfidf_des)
```




    NMF(n_components=10, random_state=1)




```python
from sklearn.decomposition import NMF
nmf_model2 = NMF(n_components=10, random_state=1)
nmf_model2.fit(dtm_with_tfidf_fea)
```




    NMF(n_components=10, random_state=1)




```python
word_list = []
probability_list = []

top_number = 10
count = 0
for probability_number in nmf_model1.components_: # model.components contains the prob of each word for each doc
    text_message = f"Top words for house {count} are : "
    print(text_message)    
    for number in probability_number.argsort()[-top_number:]: # we're only interested in the top words
        print([tfidf1.get_feature_names()[number]], end= "")
        word_list.append([tfidf1.get_feature_names()[number]])
        probability_list.append(number)
    #show_chart(word_list, probability_list, text_message)
    print("\n")  
    count += 1
```

    Top words for house 0 are : 
    ['dining']['double']['garage']['home']['family']['large']['bedroom']['rear']['room']['garden']
    
    Top words for house 1 are : 
    ['park']['area']['location']['amenities']['road']['excellent']['city']['centre']['dublin']['property']
    
    Top words for house 2 are : 
    ['room']['spacious']['communal']['bathroom']['floor']['development']['bedroom']['living']['balcony']['apartment']
    
    Top words for house 3 are : 
    ['factual']['concern']['mistakes']['early']['strongly']['wilson']['care']['leonard']['provided']['information']
    
    Top words for house 4 are : 
    ['basin']['wash']['hand']['tiled']['lighting']['room']['floor']['window']['recessed']['ceiling']
    
    Top words for house 5 are : 
    ['lounge']['heating']['right']['advised']['ft']['turn']['left']['auctioneers']['cooke']['ray']
    
    Top words for house 6 are : 
    ['point']['bedroom']['built']['wood']['carpet']['11']['10']['tiled']['amp']['floor']
    
    Top words for house 7 are : 
    ['0m']['1m']['9m']['7m']['4m']['6m']['2m']['3m']['8m']['5m']
    
    Top words for house 8 are : 
    ['intending']['behalf']['contained']['measurement']['given']['description']['shall']['vendor']['hunters']['moovingo']
    
    Top words for house 9 are : 
    ['wc']['double']['built']['wardrobes']['carpet']['bedroom']['tiled']['laminate']['wood']['flooring']
    
    

This above function for description_block may throw an error: Then increase the components to 20, and reset to 10, this is due to the random state, and when iteration starts from 0, where description_block of certain houses are None.


```python
list = []
probability = []

top = 10
counts = 0
for probability_number in nmf_model2.components_: # model.components contains the prob of each word for each doc
    text_message = f"Top words for house {counts} are : "
    print(text_message)    
    for number in probability_number.argsort()[-top:]: # we're only interested in the top words
        print([tfidf2.get_feature_names()[number]], end= "")
        list.append([tfidf2.get_feature_names()[number]])
        probability.append(number)
    #show_chart(list, probability, text_message)
    print("\n")  
    counts += 1
```

    Top words for house 0 are : 
    ['sunny']['private']['gfch']['parking']['street']['west']['south']['facing']['rear']['garden']
    
    Top words for house 1 are : 
    ['underground']['secure']['car']['management']['balcony']['designated']['parking']['floor']['space']['apartment']
    
    Top words for house 2 are : 
    ['parking']['location']['rear']['accommodation']['alarm']['oil']['fired']['heating']['gas']['central']
    
    Top words for house 3 are : 
    ['extend']['family']['planning']['house']['semi']['subject']['garage']['potential']['detached']['large']
    
    Top words for house 4 are : 
    ['family']['convenient']['accommodation']['extending']['floor']['approximately']['area']['approx']['ft']['sq']
    
    Top words for house 5 are : 
    ['dining']['open']['fully']['bathroom']['new']['modern']['living']['fitted']['room']['kitchen']
    
    Top words for house 6 are : 
    ['access']['walk']['m50']['luas']['bus']['dublin']['city']['walking']['distance']['centre']
    
    Top words for house 7 are : 
    ['guest']['sash']['bedrooms']['alarm']['doors']['upvc']['pvc']['windows']['glazed']['double']
    
    Top words for house 8 are : 
    ['end']['extended']['located']['condition']['private']['green']['location']['quiet']['sac']['cul']
    
    Top words for house 9 are : 
    ['superb']['schools']['links']['condition']['transport']['local']['location']['amenities']['excellent']['close']
    
    


```python
topic1 = nmf_model1.transform(dtm_with_tfidf_des)
topic2 = nmf_model2.transform(dtm_with_tfidf_fea)
des = []
fea = []

for index_pos in topic1:
    des.append(index_pos.argmax())
    
for index_pos in topic2:
    fea.append(index_pos.argmax())

final_df['description_block'] = des    
final_df['features'] = fea

```


```python
final_df
```




<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ad_id</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>ber_classification</th>
      <th>description_block</th>
      <th>facility</th>
      <th>features</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>...</th>
      <th>sale</th>
      <th>bungalow</th>
      <th>detached</th>
      <th>duplex</th>
      <th>end-of-terrace</th>
      <th>semi-detached</th>
      <th>site</th>
      <th>studio</th>
      <th>terraced</th>
      <th>townhouse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>996887</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>53.418216</td>
      <td>-6.149329</td>
      <td>532353.590941</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>999327</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>53.364917</td>
      <td>-6.454935</td>
      <td>532353.590941</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>999559</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>53.273447</td>
      <td>-6.313821</td>
      <td>532353.590941</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9102986</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>53.608167</td>
      <td>-6.210914</td>
      <td>532353.590941</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9106028</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>53.262531</td>
      <td>-6.181527</td>
      <td>532353.590941</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>12369815</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>35</td>
      <td>1</td>
      <td>53.342207</td>
      <td>-6.226101</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>12416011</td>
      <td>5</td>
      <td>5</td>
      <td>11</td>
      <td>8</td>
      <td>12</td>
      <td>9</td>
      <td>53.261475</td>
      <td>-6.147720</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>12232222</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
      <td>7</td>
      <td>35</td>
      <td>8</td>
      <td>53.391619</td>
      <td>-6.205157</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>11905630</td>
      <td>4</td>
      <td>4</td>
      <td>12</td>
      <td>0</td>
      <td>20</td>
      <td>5</td>
      <td>53.360578</td>
      <td>-6.183701</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>12394865</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>53.366827</td>
      <td>-6.248329</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3482 rows × 177 columns</p>
</div>



We can see from this output, that the description_block and the features columns have transformed into a categorical feature with 10 labels each, based on top keywords present in their columns.

### Here comes the image data. I directly load a .csv file which was exported from resnet50 model training

Below is a link to the resnet 50 model I used to extract features from the image dataset.
- https://colab.research.google.com/drive/1PY9uP058KPp8X_C2-OYdDvPSC7550f6G

### Including Image features trained from a ResNet 50.  Code included in seperate Notebook.


```python
image_df = pd.read_csv('colab.csv') # this was trained in an other environment and only shown here.
```


```python
final_df = image_df
```


```python
final_df
```




<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ad_id</th>
      <th>bathrooms</th>
      <th>beds</th>
      <th>ber_classification</th>
      <th>description_block</th>
      <th>features</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>surface</th>
      <th>...</th>
      <th>img_feature_2039</th>
      <th>img_feature_2040</th>
      <th>img_feature_2041</th>
      <th>img_feature_2042</th>
      <th>img_feature_2043</th>
      <th>img_feature_2044</th>
      <th>img_feature_2045</th>
      <th>img_feature_2046</th>
      <th>img_feature_2047</th>
      <th>facility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>996887</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>53.418216</td>
      <td>-6.149329</td>
      <td>532353.590941</td>
      <td>318.851787</td>
      <td>...</td>
      <td>0.346182</td>
      <td>1.348307</td>
      <td>0.000000</td>
      <td>0.005628</td>
      <td>0.000000</td>
      <td>1.184270</td>
      <td>0.127480</td>
      <td>1.594526</td>
      <td>1.213027</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>999327</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>53.364917</td>
      <td>-6.454935</td>
      <td>532353.590941</td>
      <td>318.851787</td>
      <td>...</td>
      <td>0.026714</td>
      <td>0.740785</td>
      <td>0.000000</td>
      <td>0.079070</td>
      <td>0.000000</td>
      <td>3.622611</td>
      <td>0.371412</td>
      <td>0.289070</td>
      <td>0.949369</td>
      <td>35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>999559</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>53.273447</td>
      <td>-6.313821</td>
      <td>532353.590941</td>
      <td>318.851787</td>
      <td>...</td>
      <td>0.401499</td>
      <td>2.237763</td>
      <td>0.072135</td>
      <td>0.967058</td>
      <td>0.000000</td>
      <td>1.819107</td>
      <td>0.143481</td>
      <td>0.509172</td>
      <td>0.331188</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9102986</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>53.608167</td>
      <td>-6.210914</td>
      <td>532353.590941</td>
      <td>318.851787</td>
      <td>...</td>
      <td>0.070776</td>
      <td>0.130126</td>
      <td>0.004703</td>
      <td>0.001361</td>
      <td>0.000000</td>
      <td>2.277957</td>
      <td>0.280137</td>
      <td>0.000000</td>
      <td>1.162026</td>
      <td>35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9106028</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>53.262531</td>
      <td>-6.181527</td>
      <td>532353.590941</td>
      <td>318.851787</td>
      <td>...</td>
      <td>0.116760</td>
      <td>0.202983</td>
      <td>0.000000</td>
      <td>0.143929</td>
      <td>0.000000</td>
      <td>1.624645</td>
      <td>0.112375</td>
      <td>0.376265</td>
      <td>0.638939</td>
      <td>35</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3477</th>
      <td>12369815</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>53.342207</td>
      <td>-6.226101</td>
      <td>NaN</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000778</td>
      <td>0.304676</td>
      <td>0.028688</td>
      <td>0.224259</td>
      <td>0.004548</td>
      <td>0.962055</td>
      <td>0.108555</td>
      <td>1.107376</td>
      <td>0.278461</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3478</th>
      <td>12416011</td>
      <td>5</td>
      <td>5</td>
      <td>11</td>
      <td>8</td>
      <td>9</td>
      <td>53.261475</td>
      <td>-6.147720</td>
      <td>NaN</td>
      <td>191.300000</td>
      <td>...</td>
      <td>0.046760</td>
      <td>1.183030</td>
      <td>0.000000</td>
      <td>0.080099</td>
      <td>0.000000</td>
      <td>0.218454</td>
      <td>0.126472</td>
      <td>0.296787</td>
      <td>0.084717</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3479</th>
      <td>12232222</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
      <td>7</td>
      <td>8</td>
      <td>53.391619</td>
      <td>-6.205157</td>
      <td>NaN</td>
      <td>105.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>2.081742</td>
      <td>0.011908</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.523524</td>
      <td>0.229754</td>
      <td>0.109374</td>
      <td>0.574645</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3480</th>
      <td>11905630</td>
      <td>4</td>
      <td>4</td>
      <td>12</td>
      <td>0</td>
      <td>5</td>
      <td>53.360578</td>
      <td>-6.183701</td>
      <td>NaN</td>
      <td>130.000000</td>
      <td>...</td>
      <td>0.115299</td>
      <td>1.064319</td>
      <td>0.000000</td>
      <td>0.045372</td>
      <td>0.000000</td>
      <td>0.515674</td>
      <td>0.572230</td>
      <td>0.703695</td>
      <td>0.291449</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3481</th>
      <td>12394865</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>53.366827</td>
      <td>-6.248329</td>
      <td>NaN</td>
      <td>71.000000</td>
      <td>...</td>
      <td>0.223927</td>
      <td>0.487077</td>
      <td>0.010795</td>
      <td>0.004007</td>
      <td>0.000000</td>
      <td>0.685754</td>
      <td>0.032210</td>
      <td>0.001133</td>
      <td>0.951588</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
<p>3482 rows × 2260 columns</p>
</div>




```python
final_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3482 entries, 0 to 3481
    Columns: 2260 entries, ad_id to facility
    dtypes: float64(2052), int64(208)
    memory usage: 60.0 MB
    

Now we have a really big file to train!!

# Model Training

Resplitting into Train and Test


```python
df_Train=final_df.iloc[:2982,:]
df_Test=final_df.iloc[2982:,:]
print('train size',df_Train.shape)
print('test size',df_Test.shape)
```

    train size (2982, 2260)
    test size (500, 2260)
    

Creating datasets to feed to model.


```python
train_X=df_Train.drop(['price'],axis=1)
train_Y=df_Train['price']
test_X=df_Test.drop(['price'],axis=1)
```

Import the Regressor to Train


```python
import xgboost
xgb=xgboost.XGBRegressor()
```


```python
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
```

When you observe high training accuracy, but low test accuracy, it is likely that you encountered overfitting problem. There are in general two ways that you can control overfitting in XGBoost:

1. The first way is to directly control model complexity.
    -This includes max_depth, min_child_weight and gamma.
    

2. The second way is to add randomness to make training robust to noise.
    -This includes subsample and colsample_bytree.

You can also reduce stepsize eta. Remember to increase num_round when you do so.


```python
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.20,0.15,0.1,0.05]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
```

Import Random search to find the best parameters


```python
from sklearn.model_selection import RandomizedSearchCV
```


```python
# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=xgb,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
```

This uses a 4 fold cross validation, number of cpu cores used are 4 (n_jobs = 4)


```python
random_cv.fit(train_X,train_Y)
```

    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    

    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    9.2s
    [Parallel(n_jobs=4)]: Done  64 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=4)]: Done 154 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=4)]: Done 250 out of 250 | elapsed:  4.6min finished
    




    RandomizedSearchCV(cv=5,
                       estimator=XGBRegressor(base_score=None, booster=None,
                                              colsample_bylevel=None,
                                              colsample_bynode=None,
                                              colsample_bytree=None, gamma=None,
                                              gpu_id=None, importance_type='gain',
                                              interaction_constraints=None,
                                              learning_rate=None,
                                              max_delta_step=None, max_depth=None,
                                              min_child_weight=None, missing=nan,
                                              monotone_constraints=None,
                                              n_estimators=100, n...
                                              validate_parameters=None,
                                              verbosity=None),
                       n_iter=50, n_jobs=4,
                       param_distributions={'base_score': [0.25, 0.5, 0.75, 1],
                                            'booster': ['gbtree', 'gblinear'],
                                            'learning_rate': [0.2, 0.15, 0.1, 0.05],
                                            'max_depth': [2, 3, 5, 10, 15],
                                            'min_child_weight': [1, 2, 3, 4],
                                            'n_estimators': [100, 500, 900, 1100,
                                                             1500]},
                       random_state=42, return_train_score=True,
                       scoring='neg_mean_absolute_error', verbose=5)



The parameter estimation here is done for a subset of the dataset, hence the time taken is substantial. In practice, the whole dataset is to be trained to estimate the best parameters, hence we can obtain a better result.


```python
random_cv.best_estimator_
```




    XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.1, max_delta_step=0, max_depth=5,
                 min_child_weight=4, missing=nan, monotone_constraints='()',
                 n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                 tree_method='exact', validate_parameters=1, verbosity=None)



There’s a parameter called tree_method, set it to hist or gpu_hist for faster computation.
But I use 'exact', as suggested by the best parameter estimation.

# Deploy

now we add these parameters to our regressor training model.


```python
xgb=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=4, missing=None, monotone_constraints='()',
             n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
```

### Learning


```python
xgb.fit(train_X,train_Y)
```




    XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.1, max_delta_step=0, max_depth=5,
                 min_child_weight=4, missing=None, monotone_constraints='()',
                 n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                 tree_method='exact', validate_parameters=1, verbosity=None)



### Prediction

```python
array_pred=xgb.predict(test_X)
```


```python
array_pred[:10]
```




    array([ 780835.  ,  382269.75,  454439.16,  393493.84,  271353.  ,
            351030.75,  390479.06, 1077982.2 ,  380722.4 ,  574778.06],
          dtype=float32)




```python
pred=pd.DataFrame(array_pred)
```

# Creating a Submission file


```python
pred
```




<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.808350e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.822698e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.544392e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.934938e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.713530e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>3.537192e+05</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1.001171e+06</td>
    </tr>
    <tr>
      <th>497</th>
      <td>3.957671e+05</td>
    </tr>
    <tr>
      <th>498</th>
      <td>7.106702e+05</td>
    </tr>
    <tr>
      <th>499</th>
      <td>3.261407e+05</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 1 columns</p>
</div>




```python
sub_df=pd.read_csv('Submission_file.csv') #reading old submission file
```


```python
sub_df
```




<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12373510</td>
      <td>701423.248581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12422623</td>
      <td>561985.537703</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12377408</td>
      <td>837755.712440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12420093</td>
      <td>834321.906203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12417338</td>
      <td>425684.666076</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>12369815</td>
      <td>286286.445489</td>
    </tr>
    <tr>
      <th>496</th>
      <td>12416011</td>
      <td>977124.710212</td>
    </tr>
    <tr>
      <th>497</th>
      <td>12232222</td>
      <td>425818.933065</td>
    </tr>
    <tr>
      <th>498</th>
      <td>11905630</td>
      <td>701328.471883</td>
    </tr>
    <tr>
      <th>499</th>
      <td>12394865</td>
      <td>422429.355953</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 2 columns</p>
</div>




```python
datasets=pd.concat([sub_df['Id'],pred],axis=1)
```


```python
datasets.columns=['Id','Predicted']
```


```python
datasets #new file
```




<div class="table-wrapper" markdown="block">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12373510</td>
      <td>7.808350e+05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12422623</td>
      <td>3.822698e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12377408</td>
      <td>4.544392e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12420093</td>
      <td>3.934938e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12417338</td>
      <td>2.713530e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>12369815</td>
      <td>3.537192e+05</td>
    </tr>
    <tr>
      <th>496</th>
      <td>12416011</td>
      <td>1.001171e+06</td>
    </tr>
    <tr>
      <th>497</th>
      <td>12232222</td>
      <td>3.957671e+05</td>
    </tr>
    <tr>
      <th>498</th>
      <td>11905630</td>
      <td>7.106702e+05</td>
    </tr>
    <tr>
      <th>499</th>
      <td>12394865</td>
      <td>3.261407e+05</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 2 columns</p>
</div>




```python
datasets.to_csv('Submission_file.csv')
```

Now this Submission_file is uploaded to the kaggle environment to check for performance.
