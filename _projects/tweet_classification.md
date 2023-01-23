---
layout: page
title: Tweet classification
description: a basic algorithm that detects disasters from twitter feeds
img: assets/img/projects/tweet/twt.jpg
importance: 2
category: fun
---

**TWEET CLASSIFICATION: DISASTER DETECTION**


```python
import pandas as pd
train_df = pd.read_csv('train.csv')
```


```python
print(train_df.head())
print(train_df.shape)
print(train_df.isna().sum())
```

       id keyword geo_loc                                          text_body  \
    0   1     NaN     NaN  Our Deeds are the Reason of this #earthquake M...   
    1   4     NaN     NaN             Forest fire near La Ronge Sask. Canada   
    2   5     NaN     NaN  All residents asked to 'shelter in place' are ...   
    3   6     NaN     NaN  13,000 people receive #wildfires evacuation or...   
    4   7     NaN     NaN  Just got sent this photo from Ruby #Alaska as ...   
    
       label  
    0      1  
    1      1  
    2      1  
    3      1  
    4      1  
    (7613, 5)
    id              0
    keyword        61
    geo_loc      2533
    text_body       0
    label           0
    dtype: int64
    


```python
#checking the label ratio
train_df['label'].value_counts()
```




    0    4342
    1    3271
    Name: label, dtype: int64



> **EDA**

Looks like the data has 3 independent features (**keyword,geo_loc,text_body**) that determine the **label** (dependent feature). However, the _geo_loc_ variable has about one-third of the values as NaN. It can still be used as 66% is considerably sufficient amount of data. But manually examining the data reveals the feature doesn't really provide useful information in terms of geographical location or any significant inidicators that contribute in determining the output variable. Whereas it contains text information expressed with complete use of the individual's 'freedom of speech'. An advance algorithm that can extract useful semantics and express the variable's true intention will be best to handle this type of feature, hence it is skipped in this project.   
So, there are now 2 variables determining the output;  
1. **keyword**: categorical   
2. **text_body**: text   


Using these 2 variables a baseline model to classify the tweets containing disaster intel or not will be created.

> **Pre Processing and Feature Engineering**

First, the text feature is handled for both train and test datasets

- TRAIN


```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Basic preprocessing
l=WordNetLemmatizer()

core=[]
for i in range(0, len(train_df)):
    r=[]
    r=re.sub('[^a-zA-Z0-9]', ' ',train_df['text_body'][i])
    r=r.lower()
    r=r.split()
    r=[l.lemmatize(word) for word in r if not word in set(stopwords.words('english'))]
    r=' '.join(r)
    core.append(r)

```

- TEST


```python
test_df=pd.read_csv('test.csv')
print(test_df.head())
print(test_df.shape)
print(test_df.isna().sum())
```

       id keyword geo_loc                                          text_body
    0   0     NaN     NaN                 Just happened a terrible car crash
    1   2     NaN     NaN  Heard about #earthquake is different cities, s...
    2   3     NaN     NaN  there is a forest fire at spot pond, geese are...
    3   9     NaN     NaN           Apocalypse lighting. #Spokane #wildfires
    4  11     NaN     NaN      Typhoon Soudelor kills 28 in China and Taiwan
    (3263, 4)
    id              0
    keyword        26
    geo_loc      1105
    text_body       0
    dtype: int64
    


```python
#Basic preprocessing
l1=WordNetLemmatizer()
core1=[]
for i in range(0, len(test_df)):
    r1=[]
    r1=re.sub('[^a-zA-Z0-9().]', ' ',test_df['text_body'][i])
    r1=r1.lower()
    r1=r1.split()
    r1=[l.lemmatize(word) for word in r1 if not word in set(stopwords.words('english'))]
    r1=' '.join(r1)
    core1.append(r1)
```

Now, the Categorical feature is handled for both the train and test datasets


```python
print('The number of unique keywords in training data is: ' ,len(train_df['keyword'].unique()))
print('The number of unique keywords in test data is: ',len(test_df['keyword'].unique()))
```

    The number of unique keywords in training data is:  222
    The number of unique keywords in test data is:  222
    


```python
train_df['keyword']=train_df['keyword'].fillna('no_key')
test_df['keyword']=test_df['keyword'].fillna('no_key')
```

The missing data is filled with a custom categorical label, now the different categories will be converted to numerical labels


```python
#one hot encoding
keys=pd.get_dummies(train_df['keyword'], drop_first=True)
keys2=pd.get_dummies(test_df['keyword'], drop_first=True)
```


```python
print(keys.shape)
print(keys2.shape)
```

    (7613, 221)
    (3263, 221)
    

There are 221 labels representing 222 keywords (as the final keyword will be the absense of other keys).

> **Vectorization**


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vr=TfidfVectorizer(max_features=5000,stop_words="english")
vec=vr.fit_transform(core) #training vectors
vec1=vr.fit_transform(core1) #test vectors, used to generate final output
```

Since the dimensions are huge, about 20kplus individual features, a dimensionality reduction method is used to try and reduce redundant features. Again intuition based max feature selection of 5000.


```python
xtrain=vec.toarray()
#print(xtrain.shape)
xtrain_df=pd.DataFrame(xtrain)
print(xtrain_df.shape)

```

    (7613, 5000)
    

Now, a new combined dataframe contating the categorical encoded values and the vectors is created for the training purpose


```python
# all the independent variables in this case
X = pd.concat([keys,xtrain_df],axis=1)
X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7613 entries, 0 to 7612
    Columns: 5221 entries, accident to 4999
    dtypes: float64(5000), uint8(221)
    memory usage: 292.0 MB
    


```python
# target varible creation
Y = train_df['label'].values

#cross validation
from sklearn.model_selection import train_test_split
xt,xte,yt,yte=train_test_split(X,Y,test_size=0.2,random_state=0)
```

> **Model1**


```python
from sklearn.naive_bayes import GaussianNB
model_detect=GaussianNB().fit(xt,yt)
```


```python
yp=model_detect.predict(xte)
print(yp)
```

    [1 1 1 ... 1 0 1]
    


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(yte,yp)
```




    array([[497, 389],
           [146, 491]], dtype=int64)




```python
from sklearn.metrics import accuracy_score
accuracy_score(yte,yp)
```




    0.6487196323046619



A good 64.8% accuracy is shown on the cross validation test set, However and important observation was noticed when all the 20k features were used to train the Gaussian naive bayes model, the model accuracy was significanlty low, about 61%. And when the features were further reduced by using _min_df_ and _max_df_ parameters, the accuracy went upto 75%. This suggests that, as the features increase the gaussian naive bayes model loses its performance!

> **Model2**


```python
from sklearn.naive_bayes import MultinomialNB
model2_detect=MultinomialNB().fit(xt,yt)
```


```python
yp2=model2_detect.predict(xte)
print(yp2)
```

    [0 0 0 ... 1 0 1]
    


```python
confusion_matrix(yte,yp2)
```




    array([[747, 139],
           [188, 449]], dtype=int64)




```python
accuracy_score(yte,yp2)
```




    0.7852921864740644



However, when Multinomial naive bayes model is used the accuracy was consistant for both 20k features and when feautres were reduced by almost 85%. This shows that the Multinomial naive bayes model is somewhat efficient in terms of the resources used as well the performance given

> **Intuition specific to the current problem: Disaster classification**

In the case of classifying disasters, if a normal event is classified as a disaster that wouldn't much of a problem. However, if an actual disaster is classified as a regular event, that is an actual disaster XD. So, the metric we need to focus here is the **'False negative'**. Hence, we have to evaluate the "Recall" of the model to validate the efficiency.


```python
from sklearn.metrics import recall_score
```


```python
#GaussianNB
recall_score(yte,yp)
```




    0.7708006279434851




```python
#MultinomialNB
recall_score(yte,yp2)
```




    0.7048665620094191



!! The recall for the GaussianNB model is better, meaning it is predicting lesser false negatives. This might also be due to the presence of lesser negative classes during the cross validation too. However the accuracy of MultinomialNB is far higher than the GaussianNB for any number of features. And also, both the models showed a similar recall when further hyperparameters are tuned. So, the model using least resources to produce effective results can be used according to the specific use case.

> **Other Models**


```python
from sklearn.svm import SVC
cl=SVC(kernel="rbf")
model3_detect=cl.fit(xt,yt)
```


```python
yp3=model3_detect.predict(xte)
```


```python
accuracy_score(yte,yp3)
```




    0.7839789888378201




```python
recall_score(yte,yp3)
```




    0.6671899529042387



The SVM model seems to be performing worse than the previous models, however the choice of kernel and other parameter tuning might give us a different result. When used a linear kernel, The accuracy came out to be 80% and the Recall was about 69% which wasn't too far off from the previous models, but also with a better accuracy. Hence, for the actual prediction a generalized SVM with a linear kernal can be used in this case.

> **Final Result**


```python
cl2=SVC(kernel="linear")
disaster_classifier=cl2.fit(X,Y)
```


```python
xtest=vec1.toarray()
xtest_df=pd.DataFrame(xtest)
print(xtest_df.shape)
```

    (3263, 5000)
    


```python
X2=pd.concat([keys2,xtest_df],axis=1)
X2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3263 entries, 0 to 3262
    Columns: 5221 entries, accident to 4999
    dtypes: float64(5000), uint8(221)
    memory usage: 125.2 MB
    


```python
Result=disaster_classifier.predict(X2)
```


```python
test_df['disaster_alert']=Result
```


```python
test_df.head()
```




<div>
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
      <th>id</th>
      <th>keyword</th>
      <th>geo_loc</th>
      <th>text_body</th>
      <th>disaster_alert</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>no_key</td>
      <td>NaN</td>
      <td>Just happened a terrible car crash</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>no_key</td>
      <td>NaN</td>
      <td>Heard about #earthquake is different cities, s...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>no_key</td>
      <td>NaN</td>
      <td>there is a forest fire at spot pond, geese are...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>no_key</td>
      <td>NaN</td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>no_key</td>
      <td>NaN</td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df['disaster_alert'].value_counts()
```




    0    2363
    1     900
    Name: disaster_alert, dtype: int64



The training data had about 42.9% of the tweets classified as containing info on disasters, and the test data resulted in about 27.5% of the tweets to be about disasters. Which might be closer to the actual true information as the sample size of the test data was about half of the train data, and also the linear SVM model has an accuracy about 80%. Hence, for further improvements, more feature engineering and hyperparameter tuning will be significantly helpful.

> **Conclusion**

With models based on decision trees (XGboost,Randomforst etc) the results would be much better. With these boosted techniques along with neural networks and ensemble models, the classification can be performed in an enterprise level and can even be productized. Fundamentally, better extraction techniques, feature engineering, and a solid hyperparameter tuning will mostly does the job of classification. This exercise only focuses on developing a baseline model with a decent classification ability.  
An ideal way to solve this kind of problem will be something like this: 

<div>
{% include figure.html path="assets/img/projects/tweet/ss3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

```python

```
