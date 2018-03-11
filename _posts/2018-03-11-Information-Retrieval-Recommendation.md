---
layout: post
title: Document classification with K-means 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## An example of job advertisement unsupervised classification using K-means. 

*Keywords*: Information retrieval, clustering, recommendations, Tf-IDF, classification

Imagine a user reading a book description, reading job advertisings, or looking at images of houses. We would like to recommend similar books, jobs or houses. Formally, we want to have a structured representation of our data, where we can easily navigate the feature space and find neighbouring points to be used as suggestions. 

If you are familiar with text represantation and K-means, and are more interested in the [job adverstiment classification with k-means](#JobExample), jump directly to the [example](#JobExample).



## Clustering and unsupervised classification 



<div align="left|right|center|justify">The goal of clustering is to discover groups of data that share similar features. For example, you want to find job descriptions that are similar to a description you already read. The job classification labels may not be available or may be to broad (scientist, manager, etc), and therefore useless when you want to find similar jobs. We want to structure the data (job descriptions) in our database, so that we can suggest other jobs. Moreover, in the presence of feedback, we can use it to learn preferences.  </div>



<p align="left|right|center|justify">

Looking for similar features in a dataset can be done with KD-Trees or K-nearest neighbours, for example. Clustering can be done with K-means, LDA, among others. I will cover only one deterministic clustering technique (K-means ) and one probabilistic clustering technique (LDA). </p>



### Text representation  ( a little bit of NLP)

I will not go into details of text representation or bag of words, but will only briefly highlight what seems essential to understand the features we will cluster. For the text representation we will use TF - IDF (Term frequency - inverse document frequency) to extract features from each text $$d_i$$ in our database. 


TF-IDF is a bag-of-words representation. That is, it counts the occurrences of each word in article $$d_i$$ (TF), and multiplies it by a scaling factor (IDF)  $$ \log \frac{ \# documents}{1 + \text{ #documents  using  that   word}} $$, where TF is down-weighted if it is a word appearing in every document in the database.  To be precise, I will be using a (1,2,3)-gram bag of words representation. This means I look for occurrences of single words, but also pairs and triplets of words. For example, 2-grams (bigrams) are useful to distinguish between occurrences of "applied mathematician" and "pure mathematician".  The vocabulary space is then the set of unique (1,2,3)-grams appearing in the corpus (collection of all texts). Let $$N$$ denote the size of the vocabulary. Each document $$d_i$$ is represented with a sparse vector $$x_i$$ of size $$N$$, where a non-zero entry $$j$$ of $$x_i$$,  $$x_i(j)$$ indicates the TF-IDF of vocabulary element $$j$$. It is common practice to normalise the each feature vector, to make the feature representation independent of the text length.  If there are $$L$$ documents in the corpus, the feature representation is a sparse matrix $$X$$ of dimension $$L \times N$$.



In Python, the *scikit-learn* package has bag-of-words feature extraction methods, including TF-IDF: *sklearn.feature_extraction.text.TfidfVectorizer*. This comes with various options, including $$n-gram$$ range, tokenization, stop-word removal, accent removal, normalisation type (l1,l2,none), maximum features, among others, and provides as output a sparse matrix representation.




# K means: 

K-means is a clustering algorithm that assigns a cluster label $$l_j$$ to each document $$d_i$$. This labelling is known as a hard assignment, as each document belongs to only one label $$j$$. 


Each cluster is characterised by :

- a centroid $$c_i$$

- the shape of each cluster in the feature space (The vanilla K-means, will assume all cluster are symmetrical in all dimensions). 

Denote $$z_j$$ the cluster labels, $$c_j$$ the cluster centres, and $$x_i$$ the observations.  

With clustering, the objective is to minimize the distance to the cluster centers:

$$
\phi =  \sum_{j=1}^k \sum_{x_i \in C( z_j)} d(x_i, c_j)
$$

Algorithm:

1. Initialize cluster centres. 

   $$ \mu_1,…,\mu_k$$.

2. While not converged:

   1. Assign each observation $$x_i$$ to one cluster, $$z$$:

      ​	$$z_j = \arg min_j\ \  d(c_j, x_i) $$, where  $$d(,)$$ is a distance function, i.e: $$l_2$$ norm. 

   2. Update the cluster centers with the observations assigned in previous step. 

      ​	$$ \mu_j = \frac{1}{n_j} \sum_{x_i \in C( z_j)} x_i $$
      





Because the k-means objective function is not convex, we are never guaranteed to find the global optimum. K-means will converge to a local optimum and, as any iterative optimization of a non-convex  function, the value we converge to will depend on the initial values of the cluster centres. This gives rise to **k-means++**, which is the same as k-means, but with a smart cluster initialisation.  



The algorithm for cluster center initialisation is as follows:

1. Choose the first cluster center randomly. 

2. While the number of cluster centres is less than then number of clusters:

   1. for each  $$x_i$$,  for each cluster center $$c_j$$,   

      ​    compute $$d(x_i,c_j)$$, keep the distance to the nearest cluster centre $$b_i$$ 

      ​           if $$d(x_i,c_j)<b_i$$ then $$b_i = d(x_i,c_j) $$ 

   2. Choose a new cluster center with probability proportional to $$b_i^2$$, to favour choosing a cluster center away from other clusters.

      ​



**How many clusters do we need?**

Note that until now we have assumed that the number of clusters $$k$$ is fixed. In reality, we do not know how many cluster centres are needed to best represent our data. With more centres, $$\phi$$ in equation (1) is lower, but will not generalise well. When plotting the misfit $$\phi \ Vs.  k $$,  it will typically resemble an L-curve. The best number of clusters can usually be found in the elbow of the L-curve.  



# <a name="JobExample"> Example : Job advertisement clustering </a>

If you have ever looked for jobs, you will know that keyword search is clearly not enough to make an efficient search. For example, if you search for jobs as "data scientist", you will find that the focus varies with financial applications, health related, o marketing and retail stores, in research institutions, etc.  Although you can try to refine your keyword search, it is time consuming to open and close job offerings when you realise they don't have the focus you're interested in. 



I decided to cluster the job advertisements, and depending on the most important features of each cluster, decide which job offers are more aligned with my interests. It straight forward, but it can save you a lot of time and make you more efficient in your job search. The Jupyter notebook, that runs on Python3, can be found [here](!https://raw.githubusercontent.com/whatisAI/Classification_Adds/master/Indeed_JobClassificationKmeans.ipynb). The outline is as follows:



* [Step 1](#step1): Query the job advertisement aggregator, such as Indeed, for the jobs you are interested in, in a specific city, sorted by date (most recent first). Read all the job advertisements and create a corpus.  
* [Step 2](#step2):  Do a little pre-processing: tokenise, remove stop-words and words in the indeed webpage by default, use TF-IDF to create the feature representation. 
* [Step 3](#step3):  Use a k-means algorithm. 
* [Step 4](#step4):  Look at the most important features in each cluster. 
* Step 5: Looking at the features in each cluster, retain only job offers in the cluster where you feel identified! Done! 


## <a name="step1">Step 1: Gather job advertisements from Indeed, and create a corpus. </a>

Query the job advertisement aggregator, such as Indeed, for the jobs you are interested in, in a specific city, sorted by date (most recent first). Read all the job advertisements and create a corpus. 

For the first step, I used what was done in [this great notebook](https://jessesw.com/Data-Science-Skills/ ), and only did some minor tweaks to sort by date. The main parts are to read contents from a website:

```python
site = urllib.request.urlopen(website).read()
```

and use the package *BeautifulSoup* to extract html from the site:

```python
soup_obj = BeautifulSoup(site, "lxml")
```

## <a name="step2"> Step 2: Feature extraction </a>

Do a little pre-processing: tokenise, remove stop-words and words in the indeed webpage by default, use TF-IDF to create the feature representation. 


```python
def create_features(corpus, nmin=1,nmax=3,nfeat=5000):    
    vectorizer = TfidfVectorizer(ngram_range=(nmin,nmax), min_df = 1, 
                                 sublinear_tf = True, max_features = nfeat)
    job_features = vectorizer.fit_transform(corpus)
    return vectorizer, job_features # End of the function
```

## <a name="step3"> Step 3: Use a k-means algorithm. </a>

K-means clustering call.  Specify the k-means++ initialization, number of cluster and maximum number of iterations.  For large datasets, and to make k-means scalable, Clustering can be done in batches. The two methods will not converge to the same results. In this case, because we do not have a large dataset, we do not need to use batches.

```python
def do_clustering(some_features,true_k=2, do_svd=0):
    kmeans_minibatch = 0
    if kmeans_minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=1)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=1)

    print("Clustering sparse data with %s" % km)
    km.fit(X)
    return km
```

## <a name="step4"> Step 4: Look at the most important features in each cluster.  </a>

For each cluster center, which can be accessed in *km.__dict__['cluster_centers_']*, we can look at the features that have the highest weights:

```python
def importance_features(feat_names, km, perc=99.9):
    res=km.__dict__
    for iclass in set(km.labels_):
        print('\n****** \nImportant features for class ',iclass,'\n')
        for ii,iv in enumerate(res['cluster_centers_'][iclass]):
            if iv > np.percentile(res['cluster_centers_'][iclass],perc)  :
                print('{0:<30s}{1}'.format(feat_names[ii],iv))
```





For example, searching for jobs as **data scientist** the clusters, have the following features for each class:



```python
****** 
Important features for class  0
analyst                       
quantitative                  
quantitative analyst          
research analyst              
symbasync                     

****** 
Important features for class  1
data                          
data scientist                
learning                      
machine learning              
scientist                     

****** 
Important features for class  2
analytics                     
business                      
data                          
learning                      
machine learning              
****** 
Important features for class  3
customers                     
data scientist                
pm                            
scale                         
shopping                      
****** 
Important features for class  4
development                   
employment                    
health                        
healthcare                    
support                       
****** 
Important features for class  5
developer                     
engineer                      
insurance                     
product                       
software
```



Which can be summarised as:

 **Cluster 0** : analyst, quant

**Cluster 1:**  data scientist, machine learning

**Cluster 2:** analytics, business

**Cluster 3:** customer, data science, shopping

**Cluster 4:** health, development

**Cluster 5:** developer, engineer, insurance





## Final Comments and further improvements:



- The number of clusters has been arbitrarily chosen here. Feel free to increase or decrease them, according to the level of granularity you are looking for. When plotting the misfit $$\phi \ Vs.  k $$,  it will typically resemble an L-curve. The $k$ where the curve bends is usually a good tradeoff between reducing $$\phi$$ and keeping the number of clusters small.

- Remember that every time you run k-means you will get a different clustering result. 

- This classification is far from perfect. As long as removing stop words, I removed some common names of recruiters, or words that appear in job offerings from some groups, but that remain un-informative. For example, some job offerings explain benefits, other don't. Either way, this word is not important in our classification, so I remove it. A lot more can be done in this sense. 

- K-means clustering assumes all clusters are symetrically in all dimensions (all features have the same spread). This is not necessarily true, or a good approximation. In the next post we will see how to improve on this.

- A **probabilistic labelling** makes more sense in this example. Some job offerings may intersect two categories. This will be the topic of the next post. 

  ​