---
layout: page
title: Topic modeling
description: An unsupervised clustering example project
img: assets/img/projects/topic/cluster.jpg
importance: 1
category: work
---
- Author: Mohan Ramesh

# Topic Modeling

From the 737 documents pre-classified into 5 topical groups, this notebook tries to create further interesting sub-groups within those classes (Intra-class clustering).
Manually examining the documents hinted towards having the headline of the news articles as the first line of each of the documents, and the rest being the news. The approach followed here is a simple baseline skeleton model with scope to intervene in various stages of the development process.   
- A higher level architecture of the approach followed can be visualized like this:

<div>
{% include figure.html path="assets/img/projects/topic/ss.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

>**Getting the data ready**

The news is split into its respective title and the content. Since there is only 1 feature, it can be considered as " feature engineering " step in this case and also for other reasons like;  
- most of the words in the title are already present in the content, 
- reducing redundancy,
- entity extraction. 

_**ps**: wanted to extract entities from the title (nouns), but made the process complicated:(_



```python
import os
import glob
import pandas as pd

#not the best way to load the data, but hardcoded to specific need

df= pd.DataFrame(columns=['loan'])
for filepath in glob.glob(os.path('train.txt')):
    my_file = open(filepath, 'r')
    my_text = my_file.read()
    my_file.close()
    df.loc[len(df)] = [my_text]

df2['title'] = df1['news'].apply(lambda x: ''.join(x.split('\n')[0]))
df1['news'] = df1['news'].apply(lambda x: ''.join(x.split('\n')[1:]))
df = pd.concat([df2, df1],axis=1)
df
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
      <th>title</th>
      <th>news</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Basic pre-processing is performed by using the NLTK, re libraries.


```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Basic preprocessing
l=WordNetLemmatizer()
core=[]
for i in range(len(df)):
    
    r=re.sub('[^a-zA-Z0-9()]', ' ',df['news'][i])
    r=r.lower()
    r=r.split()
    r=[l.lemmatize(word) for word in r if not word in stopwords.words('english')]  #stemming will make lose the meaning of created subtopics in this case
    r=' '.join(r)
    core.append(r)



```

> **Vectorization**

TF-IDF is chosen over a basic bag of words in hopes of getting more semantic value out of the corpus. However, the bag of words will form poor clusters anyway in this particular case as the amount of repeated words are significantly higher since the topics are already inside a class!   
**_note_**: There are advanced vectorization techniques to extract features and inturn creating more meaningful clusters.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
vr=TfidfVectorizer(max_features=350,max_df=0.70,min_df=4,ngram_range=(1,2),stop_words = "english")
vec=vr.fit_transform(core)
```

At most there were found to be around 750+ unique features in the athletics class, and top 350 have been chosen be converted into vectors, to reduce the dimensionality of the model.  
Other model parameters are selected vaguely based on intuition and a little on trial and error basis.

> **Finding optimal K-value**

Since, the amount of subtopics to generate is purely subjective to each class, running the alogrithm to create n number of groups starting from 1 is a good approach. This method is called the 'elbow method' as the ideal # of groups will be selected at the point where the graph of this K-means tend to abruptly stabalize, which looks like an elbow!


```python
from sklearn.cluster import KMeans
prom=[]
k=range(1,10)
for i in k:
    model=KMeans(n_clusters=i)
    model.fit(vec)
    prom.append(model.inertia_)
```


```python
import matplotlib.pyplot as plt
plt.plot(k,prom)
```




    [<matplotlib.lines.Line2D at 0x27d41e95730>]




<div>
{% include figure.html path="assets/img/projects/topic/output_14_1.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>


The Kmeans plot algorithm from 1 cluster to 10 clusters, looks pretty much narrow and the scope to form further subgroups seems to be also very narrow (Advanced algorithms might give better results or maybe other classes will show some cluster possibility). However, there seems to be a slight bend at k=2 and k=4 (baby elbow :))

<div>
{% include figure.html path="assets/img/projects/topic/ss1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

Forming just 2 clusters from 100+ documents seems a little monotonous, hence 4 clusters are created further to make the project interesting.


```python
true_k=4
true_model=KMeans(n_clusters=true_k)
true_model.fit(vec)
```




    KMeans(n_clusters=4)



Parameter tuning for the model is not performed here, and the results might be better with some tweaks inside the Kmeans parameters

> **Forming clusters**


```python
centroids = true_model.cluster_centers_.argsort()[:,::-1]
terms=vr.get_feature_names()
for i in range(true_k):
    print("cluster"+str(i))
    print("-------")
    for ind in centroids[i,:10]:
        print(terms[ind])
    print("\n")
```

    cluster0
    -------
    indoor
    european
    holmes
    world
    record
    second
    olympic
    best
    championship
    birmingham
    
    
    cluster1
    -------
    race
    cross
    marathon
    country
    cross country
    radcliffe
    london
    paula
    world
    chepkemei
    
    
    cluster2
    -------
    kenteris
    test
    thanou
    iaaf
    greek
    charge
    drug test
    tribunal
    drug
    athens
    
    
    cluster3
    -------
    conte
    balco
    drug
    collins
    jones
    doping
    steroid
    performance enhancing
    enhancing
    enhancing drug
    
    
    

The resulted clusters look closely related yet, a prominant pattern can be seen among all of them.

> **Final Result**


```python
clusters=true_model.predict(vec)
df['clusters']=clusters
df
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
      <th>title</th>
      <th>news</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Claxton hunting first major medal</td>
      <td>British hurdler Sarah Claxton is confident she...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O'Sullivan could run in Worlds</td>
      <td>Sonia O'Sullivan has indicated that she would ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Greene sets sights on world title</td>
      <td>Maurice Greene aims to wipe out the pain of lo...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IAAF launches fight against drugs</td>
      <td>The IAAF - athletics' world governing body - h...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dibaba breaks 5,000m world record</td>
      <td>Ethiopia's Tirunesh Dibaba set a new world rec...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pittman drops coach for UK base</td>
      <td>Australia's world 400m hurdle champion Jana Pi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Lewis-Francis shakes off injury</td>
      <td>Sprinter Mark Lewis-Francis is determined to g...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Holmes facing fine over trials</td>
      <td>Double Olympic champion Kelly Holmes faces a f...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Freeman considers return to track</td>
      <td>Former Olympic champion Cathy Freeman said she...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Event devised to nurture athletes</td>
      <td>UK Athletics has launched a new outdoor series...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 3 columns</p>
</div>



> **Visualizing clusters**

A method called 't-SNE' that visualizes high dimensional data developed by Laurens van der Maaten, an AI researcher at facebook. An attempt to adapt that technique was failed, and instead a cluster plot using PCA was made. The idea was taken from a stackoverflow article.

Link to the scientific literature: https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf   
Link to the stackoverflow article: https://stackoverflow.com/questions/27494202/how-do-i-visualize-data-points-of-tf-idf-vectors-for-kmeans-clustering 


```python
from sklearn.decomposition import PCA
kmeans_indices = true_model.predict(vec)
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vec.toarray())
colors = ["r","g","b","m","c"]
x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmeans_indices])

for i, txt in enumerate(df['title']):
    ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))
    

```


<div>
{% include figure.html path="assets/img/projects/topic/output_27_0.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>


>**Parting Thoughts**

The Athletics class had several interesting areas, but overall the subtopics with which a majority of the documents coincided involded Indoor sports, Outdoor sports and Performance enhancing drugs. The subtopics ofcourse had major events like olympics, individual records, geographic location of the event, individuals involved etc. The resulting clusters that were formed in this particular excercise were:   
**Cluster0**: Events and records from top competetions which are mainly indoor   
**Cluster1**: Outdoor and track events such as marathons and races   
**Cluster2**: Drug tests and people involved in drugs especially with the governing body IAAF   
**Cluster3**: Individuals relating to steroids and PEDs but not much overlap with IAAF. ( Maybe just accusations and no tests )   

The same model was used to test documents in other classes as well. The results were similar as the classes are closely related and generating interesting topics with just a centroid based clustering algorithm(K-means) is not ideal. The 'cricket' class showed groups of countries which participated in certain competetions and the individuals who were the highlight between those fixtures, and the suit was followed by other team sports as well (Rugby and football). However, Tennis did not show any cluster signs, and plot was scattered realtively even.

This how the **tennis** plot looked like: <div>
{% include figure.html path="assets/img/projects/topic/ss2.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
</div>
Though, clusters were created, they did not have any semantic value.

To conclude, this is one of the basic approaches to perform intra-class clustering. A higher semantic capture can be done with the use of advanced algorithms especially based on artificial neural networks.


```python

```
