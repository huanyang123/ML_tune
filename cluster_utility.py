'''
----------------------------------------------------------------------------------------------------
This is a general utility function to perform various tasks for un-superwised ML
----------------------------------------------------------------------------------------------------
By Huanwang Henry Yang  (2019-06-12)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits, make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_samples, silhouette_score
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import scipy.cluster.hierarchy as sch
import matplotlib.cm as cm
from tqdm import tqdm   #provide visual bar

#------------------------------------------------------------------------  
#------------------------------------------------------------------------  
#------------------------------------------------------------------------  
def cluster_mean_table(df_in, col_all, cluster_name='cluster') :
    '''cluster_mean_table(df_in, col_all, cluster_name='cluster')
    Generate a table for the average of all the given attributes. 
    df_in: a DF     
    col_all : the given column names to get the averaged values.
    '''
    
    df=df_in.copy()
    
    col= col_all + [cluster_name]
    df1=df[col].groupby([cluster_name]).mean().T  #DF for all attributes
    df2=df[col_all].mean().to_frame('overall_mean')  #Df for the avg of attributes
    df_12=pd.concat([df1, df2], axis=1)  #merge by column
    
    # add a row for cluster_counts
    df3=df[cluster_name].value_counts().to_frame('cluster_counts').T 
    df_w=pd.concat([df3, df_12], axis=0)
    df_w.loc['cluster_counts', 'overall_mean'] = df3.mean(axis=1)['cluster_counts']
    
    #calculate the ratio between the overall_mean and each cluster
    coln=df_w.drop(['overall_mean'], axis=1).columns
    d1=df_w.drop(coln, axis=1)
    d2=df_w[coln].div(df_w.overall_mean, axis=0)
    df_ratio=pd.concat([d2, d1], axis=1 )

    return df_ratio #, df_3

#------------------------------------------------------------------------  
def add_cluster_2df(df_file, df_fit, kmean, npca=0):
    '''add_cluster_2df(df_file, kmean)
    df_file: a DF to which the clusters are put for details analysis. 
    df_fit: is the DF for fitting KMeans. (only used here to print the score).
    kmean: is the KMeans model after fitting
    npca: add the number of component (npca) to the df_file for display (e.g. by Tableau)
    '''

    #IF PCA, create a DF for the PCA transformed data (add it to df_file for display only)
    if (npca>=2) : 
        col_pca=['pca_comp_'+str(i+1) for i in range(npca) ]
        df_pca=pd.DataFrame(data=df_fit[:,0:npca], columns=col_pca)
        #print(df_pca.shape)
    
    y_km = kmean.labels_     #same as #y_km = km.fit_predict(df3)
    df_f=df_file.copy().reset_index(drop=True)  # 

    df_f['cluster'] = y_km +1  #put the predicted cluster into the DF

    if (npca>=2) :  
        df_f=pd.concat([df_f, df_pca], axis=1)

# reorder the clusters from large to small
    arg=list(df_f.cluster.value_counts().index) #original value
    rep=list(df_f.cluster.value_counts().sort_index().index) #reordered value
    repl=['cluster_' + str(i) for i in rep]  #convert to string
    df_f.cluster.replace(arg, repl, inplace=True)

#   print('center of the cluster=\n', km.cluster_centers_)
    print('Kmeans score= {:.2f}' .format(kmean.score(df_fit)))
    print('Cluster and observation\n', df_f.cluster.value_counts().sort_index())
    
    return df_f


#------------------------------------------------------------------------  

def cluster_var_selection(df_in, col=[], maxeigval=0.8):
    '''Select variable from each cluster based on PCA, eighen value, R_Sq ratio
    RS_ratio= 1- RS_Own_cluster / RS_Next_closest_Cluster
    Select one variable from each cluster which is having minimum RS_ratio.
    
    maxeigval: given 0.8
    maxeigval = 0.8. It means that clusters will split if the second eigenvalue 
    is greater than 0.8. A larger value of this parameter gives fewer clusters 
    and less of the variations explained. Smaller value gives more clusters and 
    more variation explained. The common choice is 1 as it represents the average 
    size of eigenvalues.
    '''
    from varclushi import VarClusHi

    df=df_in.select_dtypes(include='number') #must be num
    if len(col)>0: 
        df=df_in[col]
        
    vmod=VarClusHi(df, maxeigval2=maxeigval, maxclus=None)
    vmod.varclus()
    dd=vmod.rsquare        
    return dd

#------------------------------------------------------------------------  
def kmean_test_init(X, n_clusters=3):
  '''Kmean depend on the initial allocation of centroid.
  test 5 different allocation of centroid, and pick the smaller score
  '''
  import random
  n_iter = 5
  score=[]
  for i in range(n_iter):
    # Run local implementation of kmeans
    random_num=random.randint(0, 1000)
    km = KMeans(n_clusters=n_clusters,random_state=random_num)
                
    km.fit(X)
    val=km.score(X)
    score.append(val)
    print("random_state=", random_num, "score=", val)
  return score

#------------------------------------------------------------------------  
def silhouette(X, n_clust):
  '''A function to determine the degree of separation between clusters.
     X: a np.array;  n_clust: number of the given maximum clusters.
     
1. Compute the average distance from all data points in the same cluster (ai).
2. Compute the average distance from all data points in the closest cluster (bi).
3. Compute the coefficient: (bi - ai)/max(ai,bi). Values in the interval [-1, 1]

If it is 0 –> the sample is very close to the neighboring clusters.
It it is 1 –> the sample is far away from the neighboring clusters (correct cluster).
It it is -1 –> the sample is assigned to the wrong clusters. 

--------------
The Yellowbrick in SKlearn is gives visual check.

The vertical red line is the average of the score. 
For good clustering, it should satisfy the following conditions:
1. The mortif (brick) should be above (pass) the average line. 
2. The tail on the left of each motif shows the overlap level with other motifs. 
    The samller the tail is,  the less overlap of the neighboring cluster.
3. Ideally, the size of each motif is similar.

  '''

  X=np.array(X)
  sscore=[]
  for n_clusters in range(2, n_clust):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    sscore.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([])
  # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % (i+1), alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

  plt.show()
  plt.plot(range(2, n_clust), sscore)
  plt.xlabel(r'Number of clusters ')
  plt.ylabel('average silhouette_score');
 
plt.show()

#----------------------------------------------------------------
def estimate_eps(X, hline=0.3, knn=10):
    '''estimate_eps(X, hline=0.3) : estimate eps for DBscan
    '''
    from sklearn.neighbors import NearestNeighbors
    
    neighbors = NearestNeighbors(n_neighbors=knn)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.axhline(y=hline, color='r', linestyle='--')
    plt.plot(distances)
    
#----------------------------------------------------------------
def dbscan(X, eps=0.5, min_samples=5):
    ''' dbscan(X, eps=0.3, smaples=10) :
    X: an array
    The main concept of DBSCAN algorithm is to locate regions of high density that are 
    separated from one another by regions of low density.
    Density at a point P: Number of points within a circle of Radius Eps (ϵ) from point P.
    Dense Region: For each point in the cluster, the circle with radius ϵ contains at least 
    minimum number of points (MinPts).
    '''
    from sklearn.cluster import DBSCAN
    
    db_cluster = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
    
    core_samples_mask = np.zeros_like(db_cluster.labels_, dtype=bool)
    core_samples_mask[db_cluster.core_sample_indices_] = True
    labels = db_cluster.labels_

# Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    print(pd.DataFrame({'cluster': db_cluster.labels_}).value_counts())
    
    return labels

#----------------------------------------------------------------

# Gap Statistic for K means
#https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
def gap_stat(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
# Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_
# Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    
    plt.plot(resultsdf['clusterCount'], resultsdf['gap'], linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Gap Statistic');
    plt.title('Gap Statistic vs. K');    
    return (gaps.argmax() + 1, resultsdf)

#score_g, ddf = gap_stat(df4_tran, nrefs=5, maxClusters=10)

#------------------------------------------------------------------------
def elbow (X, clusters=10, figsize=(6,6)):
  ''' A function to visulize the change of WSS with number of clusters
  X: a np.array;  clusters: the number of given clusters
  '''
  sse = []
  list_k = list(range(1, clusters))

# Run the Kmeans algorithm and get the index of data points clusters
  for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)

# Plot sse against k
  plt.figure(figsize=figsize)
  plt.plot(list_k, sse, '-o')
  plt.xlabel(r'Number of clusters ')
  plt.ylabel('Sum of squared distance (within)');
  plt.grid()
 
#----------------------------------------------------------------
def kmeans_metrics(X, clusters=15, metric='elbow'):
    ''' kmeans_metrics(X, clusters, metric='elbow')
    X: a df or np;  clusters: number of clusters 
    
    metric='silhouette' or 'calinski_harabasz'
    1. Elbow: The explained variation changes rapidly for a small number of clusters 
    and then it slows down leading to an elbow formation in the curve.
    
    2. Silhouette Coefficient:
    It tells us if individual points are correctly assigned to their clusters.
    If S(i) close to 0 means that the point is between two clusters
    If S(i) is closer to -1, then we would be better off assigning it to the other clusters
    If S(i) is close to 1, then the point belongs to the ‘correct’ cluster
    
    3. Calinski-Harabasz Index
    The Calinski-Harabasz Index is based on the idea that clusters that are (1) 
    themselves very compact and (2) well-spaced from each other are good clusters.
    ** Calinski Harabasz Index is maximized for optimized cluster.
    
    '''
    # Elbow Method for K means
    
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
# k is range of number of clusters.
    if (metric=='elbow'):
        visualizer = KElbowVisualizer(model, k=(2,clusters),  timings= True)
    else:
        visualizer = KElbowVisualizer(model, k=(2,clusters), metric=metric, timings= True)
    visualizer.fit(X)        # Fit data to visualizer
    visualizer.show()        # Finalize and render figure
    
#kmeans_metrics(df4_tran,  metric='elbow')
#kmeans_metrics(df4_tran, clusters=20, metric='silhouette')
#kmeans_metrics(df4_tran, clusters=20, metric='calinski_harabasz')

#----------------------------------------------------------------
def Davies_Bouldin_index(X, center=20):
    ''' Davies_Bouldin_index(X, center=20)
    INPUT:
        X: np or df. 
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
     
    Like silhouette coefficient and Calinski-Harabasz index, the DB index captures 
    both the separation and compactness of the clusters.This is due to the fact 
    that the measure’s ‘max’ statement repeatedly selects the values where the average 
    point is farthest away from its center, and where the centers are closest together. 
    But unlike silhouette coefficient and Calinski-Harabasz index, as DB index falls, 
    the clustering improves. 
    
    look for the small index!!
    '''

    scores = []  #get the scores
    centers = list(range(2,center))
    for center in centers:
        scores.append(get_kmeans_score(X, center))
    
    plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Davies Bouldin score')
    plt.title('Davies Bouldin score vs. K')
    
#------------------------------------------------------------------------  

def get_kmeans_score(X, center):
    
    from sklearn.metrics import davies_bouldin_score

    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)
# Then fit the model to your data using the fit method
    model = kmeans.fit_predict(X)
    
    # Calculate Davies Bouldin score
    score = davies_bouldin_score(X, model)
    
    return score

#Davies_Bouldin_index(df4_tran, center=30)

#----------------------------------------------------------------
def hierarchical(X, clusters=10, figsize=(10, 8), hline=38):
  ''' The Agglomerative Hierarchical Clustering
  X : an array;  n_clust: the number of given clusters

    begin with every point in the dataset as a “cluster.” Then find the two closest
    points and combine them into a cluster. Then, find the next closest points, 
    and those become a cluster. Repeat the process until we only have one big giant cluster.
  '''

# create dendrogram (for plotint Hierarchical Clusters)
  dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# create clusters
  hc = AgglomerativeClustering(n_clusters=clusters, affinity = 'euclidean', linkage = 'ward')
  plt.axhline(y=36, color='r', linestyle='--')

# save clusters for chart
  y_hc = hc.fit_predict(X)
    
  return y_hc

#------------------------------------------------------------------------  
def gmm(X, clusters=20):
    '''
    BIC for GMM.  pick the smallest one
    
    '''
# 
    from sklearn.mixture import GaussianMixture
    n_components = range(1, clusters)
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    score=[]
    for cov in covariance_type:
        for n_comp in n_components:
            gmm=GaussianMixture(n_components=n_comp,covariance_type=cov)
            gmm.fit(X)
            score.append((cov,n_comp,gmm.bic(df4_tran)))
            
    return score
#gmm(df4_tran, clusters=20)
#----------------------------------------------------------------
def kprototypes_cost(df, categorical=[], nclusters=3):
    '''kprototypes_cost(df, categorical=[], nclusters=3) : Elbow plot with cost (very slow!)
    categorical: a list to hold the positions of the categorical variable in DF (e.g. [1, 4])
    df:  the data frame
    nclusters: number of clusters to run
    '''
    
    from tqdm import tqdm   #provide visual bar
    import plotly.graph_objects as go
    
#Choosing optimal K value
    costs = []
    n_clusters = []
    clusters_assigned = []
   
    for num_clusters in tqdm(range(2,nclusters)):
        kproto = KPrototypes(n_clusters=num_clusters, init='Huang', n_jobs=-1, max_iter=50) 
        clusters = kproto.fit_predict(df, categorical=categorical)
        
        costs.append(kproto.cost_)       
        n_clusters.append(i)
        clusters_assigned.append(clusters)

    fig = go.Figure(data=go.Scatter(x=n_clusters, y=costs ))
    fig.show
    
    return clusters_assigned

#kprototypes_cost(dft, categorical=[6], nclusters=4)    
#----------------------------------------------------------------
def kmodes_cost(df_in, ncluster=8):
    '''kmodes_cost(df_in, ncluster=8)
    https://github.com/nicodv/kmodes/blob/master/kmodes/kmodes.py
    
    def __init__(self, n_clusters=8, max_iter=100, cat_dissim=matching_dissim,
                 init='Cao', n_init=10, verbose=0, random_state=None, n_jobs=1):
    '''
    
    from tqdm import tqdm   #provide visual bar

    cost = []
    for i in tqdm(range(1,ncluster)):
        kmode = KModes(n_clusters=i, init = "Cao", n_jobs=-1)
        kmode.fit_predict(df_in)
        cost.append(kmode.cost_)
    
    y = np.array([i for i in range(1,ncluster,1)])
    plt.plot(y,cost)   
    plt.xlabel('K')
    plt.ylabel('Cost');
#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------


