'''
=======================================================================================
A general utility module that contains functions for data exploration, visulizations,
modeling preparation, feature scaling, sampling imbalanced data, model report ...

          Henry (Huanwang) Yang  (started 2016-06-12,  modified.. )
          
=======================================================================================
Python libaries needed for using the module. 
    pandas, numpy, scipy, sklearn, matplotlib, seaborn, statsmodels
    
---------------------------------------------------------------------------------------    
How to use it?

1. import this module to your python code: 

    put the module (ml_utility.py) into a folder:  for example, tune_path='??/ML-tune/'
    Type the two sentences below
    sys.path.append(tune_path)
    import ml_utility as ut

2. use a function that is in the module:
    for example "ut.df_info(df)", df is your DataFrame. 
    
3. to get help and see the usage of all the functions, type "help(ut)"
    
---------------------------------------------------------------------------------------    

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.metrics import  roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#-----------------------------------------------------------------------------------------
# ============== below are functions for data wrangling  ===================
#-----------------------------------------------------------------------------------------

data_wrangling_info= '''

Some highlights of the handy functions
1.  df_info(), report data information in details. 
    This includes outlier detection, feature correlations, 
    data type, categorical levels, percentage data missing, question marks, duplicated rows 
    and columns, constant or low variance features. 
    
2.  df_check(), check details about your data frame and export problems if exist.

3.  auto_clean(), automatically clean the data 

4.  impute(), impute missing values using knn, XGB, mean, median

5.  select_features(), select features in various ways.

6.  plot_dist(), plot the data distribution in an easy way

7.  plot_xy(), plot x & y data in an easy way

8.  plot_outlier(), plot the data outlier in an easy way.


  '''


#-----------------------------------------------------------------------
def df_info(df, types='all'):
    '''A general function to display the basic information in the dataframe.
    
    Parameters as below:
    df: a dataframe.
    types: if not given, only display basic infor. if given 'all', it will include 
           min, max, mean, std for numerical data. But for categorical columns, it 
           gives low, high, mode and std of the unique count of each level.).  
           
    Note: There is a slight difference in calculating the std between a series.std() 
    (by 1/(N-1)) and a list.std() (by 1/(N). The df.describe() used the former! 
    '''
    
    pd.set_option('max_colwidth', 35)

    variable_name = []
    total_value = []
    total_missing_value = []
    missing_value_rate = []
    unique_value_list = []
    total_unique_value = []
    df_type = []

    if types != 'all': 
        for col in df.columns:
            dt=df[col].dtype        
            variable_name.append(col)
            df_type.append(dt)
            total_missing_value.append(df[col].isnull().sum())
            missing_value_rate.append(round(df[col].isnull().sum()/df[col].shape[0],4))
            unique_value_list.append(df[col].unique())
            total_unique_value.append(len(df[col].unique()))
        
        dic={"Variable":variable_name,  "#_Missing":total_missing_value,
             "%_Missing":missing_value_rate, "Unique_Value":unique_value_list
             , "#_Unique_Value":total_unique_value, "df_Type":df_type}   
    
        missing_df=pd.DataFrame(dic)    
        missing_df = missing_df.set_index("Variable")
        return missing_df.sort_values("#_Missing",ascending=False)
       
    
    min1=[]  # last one if it is object
    max1=[]  # top one if it is object
    mean1=[] # mode if it is object
    std1=[]
    
    df_des=df.describe(include='all')
    for col in df.columns:
        dt=df[col].dtype
        
        variable_name.append(col)
        df_type.append(dt)
#        total_value.append(df[col].shape[0])
        total_missing_value.append(df[col].isnull().sum())
        missing_value_rate.append(round(df[col].isnull().sum()/df[col].shape[0],4))
        unique_value_list.append(df[col].unique())
        total_unique_value.append(len(df[col].unique()))
        
        if  dt=='int' or dt=='float': #numerical
            min1.append(df_des.loc['min', col])
            max1.append(df_des.loc['max', col])
            mean1.append(df_des.loc['mean', col])
            std1.append(df_des.loc['std', col])
        elif dt=='O':  #object
            vc=df[col].value_counts()
            min1.append(vc.index[-1])
            max1.append(vc.index[0])
            mean1.append(df[col].mode().values[0])
            std1.append(vc.std())  
        elif dt=='<M8[ns]':  #date
            vc=df[col]
            min1.append(df[col].min())
            max1.append(df[col].max())
            mean1.append(df[col].mean())
            std1.append('NA')
            
        else:
            min1.append('NA')
            max1.append('NA')
            mean1.append('NA')
            std1.append('NA')
      
    dic={"Variable":variable_name, "#_Missing":total_missing_value,\
         "%_Missing":missing_value_rate, "Unique_Value":unique_value_list, "#_Unique":total_unique_value, \
         "df_Type":df_type, "min/low":min1,"max/high":max1,"mean/mode":mean1,"std":std1 }   
    
    missing_df=pd.DataFrame(dic)
    
    missing_df = missing_df.set_index("Variable")
    return missing_df.sort_values("#_Missing",ascending=False)

#-----------------------------------------------------------------------
def df_check(df, outlier_cutoff=3, corr_cutoff=0.9, var_cutoff=0.001):
    ''' A function to report details about the dataframe (df)
    
    Input parameters as below:
    df: an input dataframe.
    outlier_cutoff: list the outliers by IQR or above the sigma level. default=3
    corr_cutoff: list the features when correlation is > corr_cutoff.
    var_cutoff: list the features when variance is < var_cutoff
    '''
  
  #check different type of data types
    dtype=[df[x].dtypes for x in df.columns ]
    print('\n=========================Checking the data type in DF ============')
    print(set(dtype))

  #split the DF into various types
    df_num=df.select_dtypes(include='number')
    df_obj=df.select_dtypes(include='object')
    df_cat=df.select_dtypes(include='category')
    df_time=df.select_dtypes(include=[np.datetime64]) #
    
    print('\nshape of df_num=', df_num.shape)
    print('shape of df_obj=', df_obj.shape)
    print('shape of df_cat=', df_cat.shape)
    print('shape of df_time=', df_time.shape)

  #check missing values (all), return a DF
    print('\n=========================Checking the the missing values ============')
    dd1=check_missing_values(df)
    print(f'Number of columns for missing values ={dd1.shape[0]}')
    if (len(dd1)>0) : print(dd1)
           
  #check outliers (return a DF) 
    print('\n=========================Checking outliers ==================')
    '''
    dd2a=do_outliers(df_num, method='iqr', col=False, types='find',cutoff=outlier_cutoff)
    if dd2a.shape[0]>0:
        print(f'The outliers determined by {outlier_cutoff}*IQR (InterQuartile Range)')  
        print(dd2a,"\n")  
    '''
    dd2b=do_outliers(df_num, method='std', col=False, types='find', cutoff=outlier_cutoff)
    if dd2b.shape[0]>0:
        print(f'The outliers determined by mean +- {outlier_cutoff}*std (standard deviation)')  
        print( dd2b)  

  #check columns with constant values or very small variances
    print('\n=========================Checking feature variance ==================')
    const_col=find_const_columns(df, cutoff=var_cutoff)

  #check related columns using the correation
    print('\n=========================Checking correlations ==================')
    corr_col=find_corr_columns(df_num, cutoff=corr_cutoff)
    if len(corr_col)>0:
        d=df_num.corr()[corr_col]
        dd3=d[d>corr_cutoff].dropna(how='all')
        print(f'The corrlation (threshold>{corr_cutoff}) table:', '\n', dd3)
    '''
    if len(corr_col)>0:
        d=df_num.corr()[corr_col]
        dd3=d[d>corr_cutoff].dropna(how='all')
        
        print(f'The corrlation (threshold>{corr_cutoff})')
        corr_matrix = df_num[dd3.index.values].corr()
    # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=0).astype(np.bool)) 
        
        print(upper.replace(np.nan, ' '))
    '''  
#    print('\n=========================Checking VIF ==================\n')
#    print(get_vif(df_num))

    print('\n=========================Checking others ==================')
   #check if any columns containing question marks, return a DF
    dd2=check_question_mark(df)
    print(f'Number of columns containing question marks={dd2.shape[0]}')
    if len(dd2)>0: print(dd2,"\n")
        
  #check if any duplicated rows 
    ndup=len(df[df.duplicated()])
    print(f'Number of duplicated rows={ndup}')
        
  #check if any duplicated column names 
    ndp_col=df.columns[df.columns.duplicated()].tolist()  
    print(f'Number of duplicated column names={len(ndp_col)}')
    if len(ndp_col)>0 : print(f'The duplicated columns names ={ndp_col}')

#-----------------------------------------------------------------------
def auto_clean(df, corr_cutoff=0.9, cate_rename_cutoff=0.2, cate_cutoff=0.02, 
               outlier_cutoff=8, missing_cutoff=0.6, var_cutoff=0.0001):
    '''A function to clean the data frame automatically 
    
    Input parameters:
    df: the data frame to be cleaned
    
    The default threshold:
    corr_cutoff=0.9,   features are picked if correlation > corr_cutoff
    cate_rename_cutoff=0.3, level is renamed if the total count <cate_rename_cutoff*std
    cate_cutoff=0.02,      column is removed if it only has two level and count min/max <0.02
    outlier_cutoff=5,      outlier above 5 sigma will be recaped
    missing_cutoff=0.6,    column is removed if more than 60% missing
    var_cutoff=0.0001      column is removed if variance is < var_cutoff
    
    return a new DF
    '''
    
    #process categorical data (rename & reduce level)
    print('\n=============processing categorical features ============')
    df1=clean_cate(df, action='drop', cutoff=cate_cutoff,  
                   std_scale=cate_rename_cutoff, col=False, name='other')
    
    #remove columns if missing precentage value > missing_cutoff
    print('\n=============processing features with missing values============')
    df2=clean_column(df1, cutoff=missing_cutoff)
    
    #remove columns highly correlated with each other
    print('\n=============processing highly correlated features ============')
    var_corr=find_corr_columns(df, cutoff=corr_cutoff)
    if len(var_corr)>0:
        df2=df2.drop(var_corr, axis=1)
        
    #remove columns with variance < a threshold
    print('\n=============processing features of small variance============')
    var_col= find_const_columns(df2, cutoff=var_cutoff)
    if len(var_col)>0:
        df2=df2.drop(var_col, axis=1)
                
    #remove duplicated columns
    print('\n=============processing duplicated features ============')
    ndp_col=df2.columns[df2.columns.duplicated()].tolist()  
    if len(ndp_col)>0:
        df2 = df2.loc[:,~df2.columns.duplicated()]
        print(f'Removed duplicated columns {ndp_col}! ')
        
    #remove duplicated rows
    print('\n=============processing duplicated rows ============')
    ndup=len(df[df.duplicated()])
    if ndup>0:
        df2=df2.drop_duplicates()
        print(f'Removed {ndup} duplicated rows! ')
    
    #recap the outlier to outlier_cutoff sigma level
    print('\n=============processing features with outliers ============')
    df3=do_outliers(df2, method='std', col=False, types='keep', cutoff=outlier_cutoff)
    
    return df3   
        
#-----------------------------------------------------------------------
def do_outliers(df_in, method='std', col=False, types='find', cutoff=3):
    '''A fountion to find and remove outliers from the input DF
  
    Input parameters:
    df_in: a input data frame
    method: 'std' (use standard deviation) or 'iqr' (use interquartile range)
    cutoff: the threshold for finding the outliers
    types: 'find', to find outiers; 'drop', to drop outlier by cutoff; 
           'keep', assign the outlier to the maximum (cutoff*mean)
    col:    given column name. if False, check all numerical features.
    
    For the Zscore method:
    1. For each column, computes the Z-score Z=(X-mean)/std. 
    2. Take the abs value below the threshold.
    3. all(axis=1) ensures that for each row, all column satisfy the constraint.
        
    Warning: 
    1. IQR will fail, if IQR=0 (over 80% const value) for both drop/keep outlier!!
    2. If too many const, use STD with high cutoff.
    Output may not be as desired, if over 90% zeros. To avoid the problem, it's 
    neccessary to drop/keep outlies by group. For example, do group 1 (col), give 
    zcutoff=3, then do group 2 (new col) and give zcutoff=4 ...
    
    '''
    
    from scipy import stats
    
    df=df_in.copy()
    
    ''' #disabled 
    if method=='zscore' and types=='drop': #remove outliers using Zscore    
        df_num=df.select_dtypes(include='number') #must be num
        df_cat=df.select_dtypes(exclude='number') #must be num
        df_num=df_num.dropna(how='all', axis=1)

        dfn=df_num[(np.abs(stats.zscore(df_num)) < cutoff).all(axis=1)] 
        dfm=pd.concat([df_cat, dfn], axis=1, join='inner')
        return dfm
    '''
    
    if not col : 
        columns=list(df.columns)
    elif type(col) is not list:
        columns=[col]
    
    val_all=[]
    for col_name in columns:  
        ddf=df[col_name].dropna()   #get the Series  
        if len(ddf)==0 : continue
        if not pd.api.types.is_numeric_dtype(ddf) : #only check numerical
            continue
        
        ss=list(ddf.unique())  #skipe check the binaries
        if len(ss)==2 and (ss==[0,1] or ss==[0,1]):
            continue      
               
        q1 = ddf.quantile(0.25)
        q3 = ddf.quantile(0.75)
        iqr = q3-q1 #Interquartile range
        
        if method=='std':
            fence_low = ddf.mean() - cutoff*ddf.std()
            fence_high = ddf.mean() + cutoff*ddf.std()
        elif method=='iqr':
            fence_low  = q1-cutoff*iqr
            fence_high = q3+cutoff*iqr       
        
        n=0
        for val in ddf:
            if val>fence_high or val<fence_low : n=n+1
        if n >0 :
            val_all.append([col_name, fence_low, fence_high,ddf.min(), ddf.max(), n])
                        
        if types=='drop':
            df=df[(df[col_name]<fence_high) & (df[col_name]>fence_low)]
            
        elif types=='keep':  #use clip             
            df[col_name]=df[col_name].clip(lower=fence_low, upper=fence_high)

    df_col= ['col_name','fence_low','fence_high','min_val','max_val','num_outlier']  
    dfn=pd.DataFrame(val_all, columns=df_col).sort_values(by=['num_outlier'], ascending=False)
    
    if len(val_all)==0 : 
        print(f'Not found outliers under cutoff={cutoff}, method={method}') 
        
    if types=='find': return dfn
    return df

#dd=do_outliers(df, method='std', col=False, types='find', cutoff=3) 

#-----------------------------------------------------------------------
def check_question_mark(df): 
    '''A function to check question marks (which maybe the same as Null)
    '''
    d=df.apply(lambda x: x.astype(str).str.contains('\?')).sum() 
    return d[d>0].to_frame(name='number of ?')
    
#-----------------------------------------------------------------------
def clean_cate(df, action='keep', cutoff=0.02, col=False,  std_scale=0.5, name='other'):
    ''' A function to reduce categorical levels (by assigning a new 
    name to the same level when the count number is below std_scale*std.
    
    df: a dataframe.
    action:  if 'drop', remove the columns that have two levels with the ratio
             between the two levels < cutoff.  If 'keep', only rename.
    cutoff: the threshold to determine if the two-level column is drop or not.          
    col: The feature names, one item or a list of items
    std_scale: it is used to multiply the standard deviation (std) to 
                determine the level name or rename.
    name: given a different name to the level. (no useful if action='drop')
    
    Return a new DF
    '''
    
    df1=df.select_dtypes(exclude=['number']) #categorical
    df2=df.select_dtypes(include=['number']) #numerical

    if not col : 
        col=list(df1.columns)
    elif type(col) is not list:
        col=[col]
        
    dic={'feature':[], 'level_renamed':[],'final_level':[], 'count_MinMax_ratio':[], 'std':[]} 
    for x in col:   #for each feature
        vc=df1[x].value_counts()
        vc_std= vc.std()
        std_scaled=std_scale * vc_std
        d1=vc.to_frame()
        ind=d1[d1[x]<=std_scaled].index.tolist()
        df1[x]=df1[x].apply(lambda y : name if y in ind  else y) #rename 
        
        df1_count=df1[x].value_counts()
        nlevel=len(df1_count) 
        ratio=min(df1_count) / max(df1_count)
        
        rev=len(ind)
        if rev>0:
            dic['feature'].append(x)
            dic['level_renamed'].append(rev)
            dic['final_level'].append(nlevel)
            dic['count_MinMax_ratio'].append(ratio)
            dic['std'].append(vc_std)
        
    df_dic=pd.DataFrame(dic) 
    if len(df_dic)>0:
        print(df_dic)
    else:
        print(f'No level renamed to {name} at ({std_scale} * standard deviation).')
        
    cond= (df_dic.final_level<=2) &(df_dic.count_MinMax_ratio<0.02)   
    cate_feature= df_dic[cond]['feature'].unique()
    
    dfn=pd.concat([df1, df2], axis=1) 
    if action=='drop' :
        dfn=dfn.drop(cate_feature, axis=1)
    return dfn

#-----------------------------------------------------------------------
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
#-----------------------------------------------------------------------
def find_non_numeric(ss, keep_na=0):
    '''find rows excluding the numbers (2. 2.3, 4)
    
    ss: an array like or a series
    keep_na: =1, keep the np.nap in the list, =0 exclude nap
    '''
    
    if keep_na==1 :
        return ss[~ss.apply(lambda x: str(x).replace('.','',1).isdigit())]
    else:
        return ss[~ss.apply(lambda x: is_number(x))] #include np.nan
    
       
#----------------------------------------------------------
def clean_column_name(dfin, old=[], new=[]):
    '''Clean the column names. replace special characters by new
    
    df_in: the input DF
    old: the old characters
    new: the new characters. 
    example:  old=['[', ')', '/', '>'], new= ['', '', '_', '_']
    '''
    df=dfin.copy()
    if len(old) ==0 or len(old) !=len(new):
      print('Zero length of input list: old=[..], new=[..]')

    all_cols = []
    for col in df.columns:
        for i in range(len(old)):
            col=col.replace(old[i], new[i])
        all_cols.append(col)
    df.columns = all_cols
    return(df)  

#-----------------------------------------------------------------------
def clean_column(df, cutoff=0.5):
    ''' remove columns if missing value >= cutoff, or a const.
    
    input
    df: the input dataframe;  dfn, the output dataframe
    cutoff: the threshold to remove the column
    
    return a new DF
    '''
    
#    df=dfin.copy()
    col_const=df.columns[~(df != df.iloc[0]).any()].values.tolist()
    if len(col_const)>0: 
        print("\nColumns with constant variance: ", len(col_const),'\n', col_const)
    
    col_ok=df.columns[(df != df.iloc[0]).any()].values.tolist()
    df0=df[col_ok]
    
    d=df0.isna().mean()  #get a series
    df1=df0[d[d<cutoff].index.tolist()]  #new df after cut
    
    dd=d[d>=cutoff].index.tolist()
    if len(dd)>0:
        print("\nColumns with missing values >", 100*cutoff,'%  :', len(dd), "\n", dd)
  
    return df1
#-----------------------------------------------
def check_missing_values(df):
  ''' A function to check any missing values, return a DF
  '''

  d=df.isna().any() #only check missing part
  d1=df[d[d].index].isna().sum()
  d2=df[d[d].index].isna().mean()
  dd=pd.DataFrame({'number_missing':d1.values, 'percent':d2*100})
  dd.sort_values(by=['percent'], ascending=False, inplace=True)

  return dd
    
#-----------------------------------------------------------------------
def find_corr_columns(df,  cutoff=0.9):
    '''A function to find the correlated columns . 
    
    input parameters
    df: a numerical dataframe or np
    cutoff: the given threshold. If >0.9, pick the column
    
    return a list of correlated features
    '''

    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than  threshold
    to_drop = [column for column in upper.columns if any(upper[column] > cutoff)]
    print(f'Features with correlation > {cutoff} is {to_drop}')
  
    return to_drop
    
#---------------------------------------------------------------------------
def find_const_columns(df, cutoff=0.00001):
    '''A function to find constant column or variance <var_threshold
  
    df: a numerical DF 
    cutoff : A threshold, if given 0, it is a constant column;  
    return a list of features ready to drop    
    '''
    
    from sklearn.feature_selection import VarianceThreshold
  
    col_const=df.columns[~(df != df.iloc[0]).any()].values.tolist()
    print("\nColumns with constant variance=", col_const)
  
  #further check if any column has variance < var_threshold
    df1=df.loc[:, (df != df.iloc[0]).any()]  #get DF with non-const columns
    df_num=df1.select_dtypes(include='number')
    select=VarianceThreshold(cutoff)
    df2=select.fit_transform(df_num)

    feature_idx = select.get_support()
    feature_name = df_num.columns[feature_idx]
    to_drop=[x for x in df_num.columns.values if x not in feature_name.values]
    print("Columns of variance <", cutoff,':', to_drop)
    col_all=to_drop + col_const
    return col_all
    
#-----------------------------------------------------------------------
def colum_length(df,  col_name):
    #count the document length. return a df
    d=df[col_name].apply(lambda x: len(x)).values
    return pd.DataFrame({'column_len':d}).column_len.value_counts()

#--------------------------------------------------------------------
def get_common_df(df1, df2) :
    '''get_common_df(df1, df2)
    combine two DF (by index), return the common columns 
    '''    
    
    col_len=len(list(set(list(df1.columns)) & set(list(df2.columns)))) 
    idx_len=len(list(set(list(df1.index)) & set(list(df2.index)))) 
    
    dd=pd.DataFrame({'column_length':[len(df1.columns),len(df2.columns), col_len], 
                 'index_length':[len(df1.index),len(df2.index), idx_len]}
                 , index=['DF1', 'DF2', 'DF1&DF2'])
    print(dd)
    dfm=pd.concat([df1, df2], axis=1, join='inner')
    dfm = dfm.loc[:,~dfm.columns.duplicated()]
    return dfm
#    list1=list(set(df1.columns) - set(df2.columns))
#-----------------------------------------------------------------------
def check_target(df, col='', target=''):
    '''A function to check how the target is distributed in each level
    
    Parameters as below:
    df: a dataframe.
    col: name of the columns
    target: name of the target

    '''
    
    if len(col)==0 or len(target)==0:
        print('You need to give both column name and target name.\n')
        return
    
    return pd.crosstab(df[target], [df[col]]).T
    
    #below is different solution
    nclass=list(df[target].unique())
    n=len(nclass)
    
    dd=df[df[target]==nclass[0]][col].value_counts().to_frame(name='target_'+nclass[0])
    for i in range(1, n):
        d1=df[df[target]==nclass[i]][col].value_counts().to_frame(name='target_'+nclass[i])
        dd=dd.merge(d1, how='outer', left_index=True, right_index=True)
        
    return dd 


#-----------------------------------------------------------------------
      
#---------------------------------------------------------------------------
def impute(df_in, var_predict='', target='', method='xgb', 
           features=[], nfeature=10):
    '''A function to impute missing values using knn, XGB, mean, median

    df_in: the input dataframe 
    method: input 'mean' & 'median' for numerical; 
            'predict' to use XGB to predict both types of missing values
    target:  The feature to be filled or predicted. if not given, do all !!
    features:  If given, will use it for prediction. if not, use all df    
    return the final DF
    
    Note; If using mean/median/xgb, the categorical is auto-imputed!
    
    '''
                
    if method.lower()=='mean' or method.lower()=='median': #the simple way
        print(f'Imputing the NULL using the {method}')
        dfn=impute_simple(df_in, method=method)
        return dfn
    elif method.lower()=='knn' or method.lower()=='xgb':
        if len(target)==0:
            print('SKIP. When using KNN or XGB, target must be given!')
            return

        if method.lower()=='knn': print(f'Imputing the NULL using KNN. VERY SLOW!!')
        if method.lower()=='xgb': print(f'Imputing the NULL using XGB. SLOW for big data!')
        if len(var_predict)>0 : 
            print(f'Imputing NULLs in {var_predict} using the {method}.')
            dfn=impute_pred(df_in, var_predict=var_predict, target=target, method=method, 
                            features=features, nfeature=nfeature)
            return dfn
        else: #
            print(f'Imputing all NULLs in the DF using the {method}.')
            dfn=df_in.copy()
            df=check_missing_values(dfn)
            col_nan=df.index.tolist()
            for col in col_nan:
                miss=df.loc[col, 'percent']
                if miss>60:
                    print(f'Skipping {col}. The missing percent {miss} > 60%')
                    continue
                dfn=impute_pred(dfn, var_predict=col, target=target, method=method, 
                                features=features, nfeature=nfeature)
            return dfn
        

#---------------------------------------------------------------------------
def impute_pred(df_in, var_predict='', target='', method='xgb', features=[], nfeature=10):
    '''A function to predict the missing values by xgb/knn. 

    df_in: The input data frame
    var_predict : The feature name (variable) to be predicted. 
    target:  The target name used for classification/regression (used for feature selection).
    method: use 'knn' for KNNImputer, or 'xgb' for XGB. If give 'knn', only predict 
            numerical values. If 'xgb', predict both numerical and categorical
    features: if given, use them for prediction
    nfeatures:  select number of best features. default 10 for prediction.
    
    Return a new DF
    
    Note: Much slower than df.fillna(df.mean()) for numerical or the
        df.fillna(df.mode().iloc[0]) for categorical. 
    '''
    
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.impute import KNNImputer
#    from sklearn.metrics import r2_score, mean_squared_error
    
    if len(var_predict)==0 or len(target)==0: 
        print("Input wrong. You need to provide target and var_predict name.")
        return
    
    df=df_in.copy()
    if len(features)>0 : df=df_in[features]

    #separate the predicted variable with the rest features 
    X=df.drop([var_predict], axis=1) 
    y=df[var_predict]  #use original
    
    #give a new DF and reduce the features to make it fast
    dfn=pd.get_dummies(X, drop_first=True) #handle possible dummy
    dfn=dfn.dropna(axis=1)   #remove columns with NULL
    dfn_s=dfn
    if(dfn.shape[1]>=nfeature): 
        yval=dfn[target]
        if yval.dtypes=='O':
            dfn_s=select_features(dfn, yval, types='Kbest', 
                                     method='class', nfeature=nfeature)
        else:
            dfn_s=select_features(dfn, yval, types='Kbest', 
                                     method='reg', nfeature=nfeature)            
    dfn_s[var_predict]=y #include the var for prediction
    

    if method.lower()=='knn': #slow
        imputer = KNNImputer(n_neighbors=4, weights="uniform")
        ddf=imputer.fit_transform(dfn_s)   #as numpy  
        dfn=pd.DataFrame(ddf, index=dfn_s.index, columns=dfn_s.columns) 
        df[var_predict]=dfn[var_predict]
        return df
    
    
    #separate rows with/without nan values
    df1=dfn_s.loc[~dfn_s[var_predict].isna()]  #a DF without nan values
    df1NA=dfn_s.loc[dfn_s[var_predict].isna()]  #a DF with target exist NaN

    #  use non-NULL rows for training and NULL row for prediction
    Xtrain=df1.drop([var_predict], axis=1) 
    ytrain=df1[var_predict]
    Xtest=df1NA.drop([var_predict], axis=1) 
    ytest=df1NA[var_predict]
    
    data_type=ytrain.dtypes
    ss=''
    if data_type=='O':
        mod=XGBClassifier()  #use XGB, no need to do encode  
        ss='XGBClassifier'
    elif data_type=='int' or data_type=='float':
        mod=XGBRegressor()  #use XGB, no need to do encode
        ss='XGBRegressor'
    print(f'Target {var_predict}, data type={data_type}, NULL predicted by XGBoost.')    
    print(f'new DF without NULL {df1.shape}, new DF with NULL {df1NA.shape}', '\n')
    
        
    mod.fit(Xtrain, ytrain)  #fit the model 
    y_pred = mod.predict(Xtest)
    
    #use the initial DF to insert the predicted values
    df2=df.loc[~df[var_predict].isna()]  
    df2NA=df.loc[df[var_predict].isna()] 
    df2NA[var_predict]=y_pred
#    print(f'Fitting R2_score={r2_score(ytest, y_pred)}')
    df_final=pd.concat([df2, df2NA], axis=0)
    
    return df_final


#---------------------------------------------------------------------------
def impute_simple(df_in, method='mean'):
    '''A simple function to impute missing value 
    
    df_in: the input dataframe 
    method: input 'mean' & 'median' for numerical; 'mode' for categorical
    
    return a new DF
    
    '''

    df_cat=df_in.select_dtypes(exclude=['number']) #categorical
    df_num=df_in.select_dtypes(include=['number']) #numerical
          
    if len(df_cat)>0 :
        df_cat = df_cat.fillna( df_cat.mode().iloc[0] )  
        
    if method =='median' :
        df_num = df_num.fillna(df_num.median())
    elif method =='mean' :
        df_num = df_num.fillna(df_num.mean())
        
    dfn=pd.concat([df_num, df_cat], axis=1)         
    return dfn
#---------------------------------------------------------------------------
#this is another impute test
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class imputer1(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy in ['mean','median']:
           if not all(X.dtypes == np.number):
               raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       elif self.strategy == 'fill':
           if type(self.fill) is list and type(X) is pd.DataFrame:
               self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
       return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)

#-----------------------------------------------------------------------------------------
# ============== below are functions for modeling   ===================
#-----------------------------------------------------------------------------------------

modeling_info='''

'''
##-----------------------------------------------------------------------
def split_data(df_in, size=0.25, target='', stratify='yes'):
    ''' randomly split the data into training and testing part
    
    df_in: the input DF
    size: the percentage of the testing part
    target: the target name in the DF
    stratify: 'yes' force to split the needed distribution (good for imbalanced data). 
    
    '''
    
    df_final=df_in.copy()
    if len(target)==0: 
        print("Error: please give the target (class) name.")
        return 

    X=df_final.drop([target], axis=1)
    y=df_final[target]
    if stratify=='yes':
        X_train0, X_test0, y_train0, y_test0 = \
        train_test_split(X, y, test_size=size, random_state=42,stratify=y)
    else:
        X_train0, X_test0, y_train0, y_test0 = \
        train_test_split(X, y, test_size=size, random_state=42)
        
    print (f'shape of X={X.shape}, X_train={X_train0.shape},  X_test={X_test0.shape}' )
    
    return X_train0, X_test0, y_train0, y_test0

#-----------------------------------------------------------------
def get_vif(df):
    '''A function to calculate VIF (variance_inflation_factor)
    
    df: the input DataFrame
    return a DF with futures and VIF
    '''
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X=df.select_dtypes(include=['number']) 
    
    df_vif = pd.DataFrame()
    df_vif["feature"] = X.columns
    vif_val=[variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    df_vif["VIF"] = vif_val
    dfn=df_vif.sort_values(by = 'VIF', axis = 0, ascending=False, inplace=False)
    
    return dfn
      
    #below for generic calculation (for reference only)
    df_vif = pd.DataFrame(columns = ['feature', 'VIF'])
    x_names = X.columns
    for i in range(0, x_names.shape[0]):
        y = X[x_names[i]]
        x = X[x_names.drop([x_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        df_vif.loc[i] = [x_names[i], vif]
    dfn=df_vif.sort_values(by = 'VIF', axis = 0, ascending=False, inplace=False)
    return dfn

#-----------------------------------------------------------------
def data_scalor(df_train, df_test='', data_type='df', scale_type='standard'):
    ''' A general function to use only the training set to scale/normalize data, 
    then apply the same transform to the test set. (no 'leaking' info.)
    
    df_train: a traing dataframe or a numpy
    df_test: a testing dataframe or a numpy
    data_type: a return data type. 'df': a dataframe (default), 'np': a numpy
    scale_type: 'standard', 'minmax', 'robust'
    
    Stochastic Gradient Descent and any distance-related calculations are sensitive to  
    feature scaling, therefore, it is highly recommended to scale the data before modeling. 
            
    1. Use MinMaxScaler (X-Xmin/(Xmax-Xmin)), if you transform a feature. 
        It's non-distorting. sensitive to outliers, value between [0,1] or [-1,1].
        good for image processing (0-255) & neural network
    2. Use RobustScaler (IQR) if you have outliers and want to reduce their influence. 
       However, you might be better off removing the outliers, instead.
    3. Use StandardScaler ((X-mean)/std ) if you need a relatively normal distribution.
    '''
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if(not (isinstance(df_train, pd.DataFrame) or isinstance(df_train, np.ndarray))):
        print("Error: input data type must be DF or Numpy.")
        return df_train, df_test
        
    #Standard: mean=0, std=1; good for clustering and PCA
    if (scale_type.lower()=='standard') :  
        scaler = StandardScaler()
    elif (scale_type.lower()=='minmax') : 
        scaler = MinMaxScaler()
    elif (scale_type.lower()=='robust') : 
        scaler = RobustScaler()
    else:
        print("Error: Given wrong scalor. Use one of (Standard, MinMax, Robust)." )
        return df_train, df_test
        
    scaler.fit(df_train)  # fit only on training data!!! (get mean and std)
    df_train_s = scaler.transform(df_train) #expect sum()=0 for each column
    
    if len(df_test)==0: #only one DF
        if(isinstance(df_train, pd.DataFrame) and data_type=='df'):
            df_train_s=pd.DataFrame(df_train_s, index=df_train.index, columns=df_train.columns)
        return df_train_s
        
    df_test_s = scaler.transform(df_test)  #use mean & std from fitting
    
    if(isinstance(df_train, pd.DataFrame) and data_type=='df'):
        df_train_s=pd.DataFrame(df_train_s, index=df_train.index, columns=df_train.columns)
        df_test_s =pd.DataFrame(df_test_s,  index=df_test.index, columns=df_test.columns)
        
    return df_train_s, df_test_s
    
#----------------------------------------------------------------------
def add_dummy_scale(X, dtype='df'):
  '''A function to scale the features & assign dummy variables to categorical.
  X: a dataframe. Note: not include target (not need to be scaled)
  dtype: if 'df', return a DF;  if 'np' , return a num array 
  '''

  #1. separate the categorical and numerical
  X_cat=X.select_dtypes(include=['object'])
  X_num=X.select_dtypes(include=['number'])

  if(X_cat.shape[1]!=0): # categorical exist
    Xn_num=data_scale(X_num, 'df')  #standerize the num data
    df_dummy=pd.get_dummies(X_cat, drop_first=True)
    Xn=df_dummy.join(Xn_num)
    if(isinstance(X, pd.DataFrame) and dtype=='df'): #return a DF
        X1=pd.DataFrame(Xn, columns=Xn.columns.values)
        return X1
    return np.array(Xn)
  else:
    Xn_num=data_scale(X_num, dtype)
    return Xn_num

#----------------------------------------------------------------------
def resample_imbalanced_data(X, y, types, ratio, target):
    ''' A function to handle imbalanced data for classification
    X: the training data with more than one features.
    y: the training data for the class only. It is a vector
    types: the model name
    target: the class 0 or 1
    ratio: the ration between class 0 or 1
    '''    
    prog=['NearMiss','RandomUnderSampler','RandomOverSampler','SMOTE','ADASYN','SMOTEENN','SMOTETomek']
    from imblearn.under_sampling import NearMiss
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN 
    from imblearn.combine import SMOTEENN
    from imblearn.combine import SMOTETomek
    random_state=0
    
    if(types.upper()=='OVER'):
        #upper-sample the majority class(es) by picking samples at random with replacement.
        rs = RandomOverSampler(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    elif (types.upper()=='UNDER'):
        #under-sample the majority class(es) by picking samples at random with replacement.
        rs = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    elif (types.upper()=='NEARMISS'):
        # define the undersampling method
        #https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
        undersample = NearMiss(version=1, n_neighbors=3)
        X_rs, y_rs = undersample.fit_resample(X, y)   # transform the dataset
                
    elif (types.upper()=='ADASYN'):
        rs = ADASYN( sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    elif (types.upper() =='SMOTE'):
        #Synthetic Minority Oversampling Technique
        rs = SMOTE(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    elif (types.upper() =='SMOTEENN'):
        rs = SMOTEENN(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    elif (types.upper() =='SMOTETOMEK'):
        rs = SMOTETomek(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    else:
        print("Error: Giving wrong argument!")
     
    
    XX=pd.DataFrame(X_rs, columns=X.columns.values)
    yy=pd.DataFrame(y_rs, columns=[target])

    perc=100.* len(X_rs)/len(X)
    print("Prog: {};  # of original X={}; # of resampled X={}; perc={:4.2}%".
           format(types, len(X),len(X_rs),perc))
    print('After Sampling, the shape of X: {}'.format(X_rs.shape))
    print("After Sampling, counts of label '1': {}".format(sum(y_rs==1)))
    print("After Sampling, counts of label '0': {}".format(sum(y_rs==0)))
        
    return XX, yy  #as dataframe    

#--------------------------------------------------------
def smote_pipeline(model, X_train, y_train):
  import warnings
  # ignore all caught warnings
  with warnings.catch_warnings():warnings.filterwarnings("ignore")

  weights=np.linspace(0.005, 0.8,10)

  pipe=make_pipeline(SMOTE(),model)
  gscv=GridSearchCV(estimator=pipe,param_grid={'smote__ratio': weights},
                    scoring='f1', cv=3)
  gs_res=gscv.fit(X_train, y_train)
  print("Grid serach best_params_:", gs_res.best_params_)
  ddf=pd.DataFrame({'score' : gs_res.cv_results_['mean_test_score'],
                   'weight': weights})
  ddf.plot(x='weight')
  return gs_res.best_params_['smote__ratio']

#model=XGBClassifier()
#model=RandomForestClassifier()
#best_ratio=smote_pipeline(model, X_train_sm, y_train_sm)


#--------------------------------------------------------

def get_feature_importance(model, feature_name):
    df = pd.DataFrame({'feature': list(feature_name),
                       'importance': model.feature_importances_}).\
                        sort_values('importance', ascending = False)
    print(df.head(len(feature_name)))

#--------------------------------------------------------
def select_feature_var(df_in, cutoff=0.05):
    ''' A function to select features based on VarianceThreshold
    This algorithm looks only at the features (X), not the desired 
    outputs (y), and can thus be more usefull for unsupervised learning.
    
    df: the input data frame.
    cutoff: the threshold to remove the column
    '''
    from sklearn.feature_selection import VarianceThreshold
    
    df=df_in.copy()
    selector = VarianceThreshold(cutoff)
    selector.fit(df)
    return df[df.columns[selector.get_support(indices=True)]]

#--------------------------------------------------------
def transf(select, X_train, y_train, X_test, y_test):
    '''
    '''
    select.fit(X_train, y_train)
    X_train_s=select.transform(X_train)
    X_test_s=select.transform(X_test)
    print( 'The shape of X_train = ', X_train.shape)
    print( 'The shape of X_train_select = ', X_train_s.shape)

    return X_train_s, X_test_s
    
#--------------------------------------------------------
def select_features(X, y, types='KBest', method='reg', nfeature=1):
    '''A funtion to select the best features to reduce noise, overfit
    
    input parameters:
    X : the training data (a data frame with n features).
    y : the training target (a data frame with one feature).
    nfeature:  the number of output features to select (like 20)
    types: (KBEST|RFE_RF|RFE_XGB|MODEL_RF|MODEL_XGB) for selection. 
    method: 'REG' for regression; 'CLASS' for classification
    
    Return a DF with the selected features.
    
    Note:
    RFE removes least significant features over iterations. So basically it 
    first removes a few features which are not important and then fits and 
    removes again and fits. It repeats this iteration until it reaches a 
    suitable number of features.

    SelectFromModel (Model Based selection) is a little less robust as it just 
    removes less important features based on a threshold given as a parameter. 
    There is no iteration involved.
    (use all features, remove those with importance below a threshold)
    
    SelectKBest: A method to select the features according to the k highest score.
    
    Warning: impurity-based feature importances can be misleading for high 
    cardinality features (many unique values)

    '''

    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, f_regression    
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn.ensemble import RandomForestClassifier 
    from xgboost import XGBClassifier, XGBRegressor

    from math import isnan


    ftype=types.upper()
    print(f"Feature selection:  method={method};  type={types};  nfeature={nfeature}")
    
    if method.upper()=='REG' :  #for regression        
        if ftype=='KBEST':  
            model = SelectKBest(score_func=f_regression, k=nfeature)        
        elif ftype=='RFE_RF':  #Recursive Feature Elimination (RFE)
            model = RandomForestRegressor(n_jobs=-1)
            select = RFE(model, nfeature)          
        elif ftype=='RFE_XGB':  #Recursive Feature Elimination (RFE)
            model = XGBRegressor(n_jobs=-1)
            select = RFE(model, nfeature)
        elif ftype=='MODEL_RF':  #model RF based method, threshold=None
            model=RandomForestRegressor(n_jobs=-1)  #use all cores
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)            
        elif ftype=='MODEL_XGB':  #model XGB based method
            model=XGBRegressor(n_jobs=-1)
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)            
        else:
            print('Error: input wrong types! (KBEST|RFE_RF|RFE_XGB|MODEL_RF|MODEL_XGB)\n')
            return
    elif method.upper()=='CLASS' : #for classification        
        if ftype=='KBEST': 
            model = SelectKBest(score_func=chi2, k=nfeature)        
        elif ftype=='RFE_RF': 
            model = RandomForestClassifier(n_jobs=-1)
            select = RFE(model, nfeature)          
        elif ftype=='RFE_XGB': 
            model = XGBClassifier(n_jobs=-1)
            select = RFE(model, nfeature)
        elif ftype=='MODEL_RF':  
            model=RandomForestClassifier(n_jobs=-1)  #use all cores
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)            
        elif ftype=='MODEL_XGB':  
            model=XGBClassifier(n_jobs=-1)
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)            
        else:
            print('Error: input wrong types! (KBEST|RFE_RF|RFE_XGB|MODEL_RF|MODEL_XGB)\n')
            return
    else:
        print('Error: Input wrong method (REG|CLASS) \n')
        return
    
    if ftype in ['KBEST'] : # different from other models        
        fit=model.fit(X, y)
        dic=dict(zip(X.columns, fit.scores_))
        dic = {k: dic[k] for k in dic if not isnan(dic[k])}
        sorted_list = sorted(dic, key=dic.get, reverse = True)

        feature = []; scores = [];
        for i in range(0, nfeature):
            feature.append(sorted_list[i])
            scores.append(dic[sorted_list[i]]) 
        dd=pd.DataFrame({'feature':feature, ' importance':scores})
        print(dd)    
        
        #if not care about the feature importance, use fit_transform
        #fit=model.fit_transform(X, y) 
        #dfn=X[ X.columns[model.get_support(indices=True)] ]
        
        return X[dd.feature.unique()]  #return DF
    else:
        select.fit(X, y)
        X_tran=select.transform(X)
        #the transf gives a array. I want it to be a dataframe (keep selected features)
        
        feature_idx = select.get_support()
        feature_name = list(X.columns[feature_idx])
        X_tran=pd.DataFrame(X_tran, columns=feature_name, index=X.index)  #make a df
        
        model.fit(X_tran, y)        
        dft=pd.DataFrame({'feature': feature_name,'importance': model.feature_importances_})
        dfn = dft.sort_values('importance', ascending = False)
        print(dfn.head(len(feature_name)))
        
        return X_tran[dfn.feature.tolist()]

#--------------------------------------------------------
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

#--------------------------------------------------------
def get_dataset_as_df(sk_dataset):
    ''' 
    https://scikit-learn.org/stable/datasets/index.html
    There are 6 toy datasets and 9 real word datasets. USE import sklearn.datasets as data
    load_(boston, iris, diabetes, digits, linnerud, wine, breast_cancer)
    Note: digits, linnerud must be treated differently
    sk_dataset : an obj for the data: example datasets.load_boston()
    '''    
    df = pd.DataFrame(sk_dataset.data, columns=sk_dataset.feature_names)
    df['target'] = pd.Series(sk_dataset.target)
    return df

#df = get_dataset_as_df(data.load_diabetes())

#--------------------------------------------------------

def evaluate_threshold(model, X_test, y_test, threshold=0.5):
    '''
    define a function that accepts a threshold and prints sensitivity and specificity
    IMPORTANT: first argument is true values, second argument is predicted probabilities
    store the predicted probabilities for class 1
   '''

    from sklearn import metrics

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

    print ('\nSensitivity:', tpr[thresholds > threshold][-1])
    print ('Specificity:', 1 - fpr[thresholds > threshold][-1])

#evaluate_threshold(model,X_test, 0.5)

#----------------------------------------------------------------------------
def compare_test_pred_class(model, X_test):
    #compare the predicted with the testing data
    #the predicted class is the one with highest mean probability estimate

    prob=model.predict_proba(X_test)
    y_pred_prob = np.argmax(prob, axis=1)  #get the index for max prob
    print(prob.shape)

    probn=[]
    for i in range(len(prob)):  
       probn.append(prob[i][y_pred_prob[i]])
    dd=pd.DataFrame({"y_test":y_test, "y_pred": y_pred,  
                     "diff":(y_pred-y_test), "probability":probn}).reset_index()

    return(dd)


#---------------------------------------------------------------------------
#Classification results
def write_result_class(X_test, y_test, y_pred, model):
    '''X_test: test set containing features only
       y_test: test set containing target only
       y_pred: the predicted values corresponding to the y_test
       model:  the model used to train the data (X_train)
       
       Note: By default, the threshold is 0.5. (If prob>0.5, y_pred=1). 
       y_pred1 = (model.predict_proba(X_test_s)[:,1] >= 0.5).astype(int)
    '''


    #for i in range(len(y_pred)): print( 'predicted, target=', y_pred[i],y_test.values[i])

    print( '\nConfusion_matrix=\n', confusion_matrix(y_test, y_pred))
    print( 'Classification_report=\n', classification_report(y_test, y_pred))

    if len(np.unique(y_test))>2 : return

    prob=model.predict_proba(X_test)[:, 1]
#    model_roc_auc = roc_auc_score(y_test, y_pred)
    model_roc_auc = roc_auc_score(y_test, prob)
    print( 'Classification accuracy=', model.score(X_test, y_test))
    print( 'Classification AUC_ROC= ', model_roc_auc)
    

    fpr, tpr, thresholds = roc_curve(y_test, prob)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC_ROC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
   # plt.savefig('AUC_ROC')
    plt.show()

##---------------------------------------------------------
def write_result_regr(y_pred, y_test, model):
    '''a function to write basic information for regression
    
    input parameters
    y_pred: the predicted Y value
    y_test: the actual Y value
    model: the algorithm used for modeling
    '''
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2=r2_score(y_test, y_pred)
    MAE= mean_absolute_error(y_test, y_pred)
    MSE= mean_squared_error (y_test, y_pred)
    RMSE=np.sqrt(MSE)
    
    print( '\nUsing %s for prediction' %model)
    print( 'r2_score = {:.2f}'.format(r2))
    a,b=y_test.mean(),y_pred.mean()
    print( 'mean values:  actual={:.2f};  predicted={:.2f}'.format(a,b))
    MAPE=100*((y_test-y_pred)/y_test).sum()/len(y_test)
    print( 'MAE={:.2f} ;  RMSE={:.2f} ;   MAPE={:.2f}%'.format(MAE, RMSE, abs(MAPE)))

#----------------------------------------------------------
#Classification results
def write_result_cnn(X_test, y_test, model):
    '''X_test: test set containing features only
       y_test: test set containing target only
       y_pred: the predicted values corresponding to the y_test
       model:  the model used to train the data (X_train)
    '''

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

# different from the regular ML. For CNN, make it 1D series to match y_test
    y_pred = model.predict(X_test)[:,0]  
    y_pred_prob=model.predict_proba(X_test)[:,0]
    # make the pred as int, ready for confusion matrix
    yy_pred=(y_pred+0.5).astype(int)

    print( '\nConfusion_matrix=\n', confusion_matrix(y_test, yy_pred))
    print( 'Classification_report=\n', classification_report(y_test, yy_pred))

    #get the model accuracy
    cnn_accuracy=model.evaluate(X_test, y_test)
    print( 'Loss & accuracy=', cnn_accuracy)

#    model_roc_auc = roc_auc_score(y_test, yy_pred)
    model_roc_auc = roc_auc_score(y_test, y_pred_prob)
    print( 'Classification AUC_ROC= ', model_roc_auc)

    #plot the AUC_ROC curve
    plot_roc(y_test, y_pred_prob, model_roc_auc)
    
#----------------------------------------------------------
def encode(df_in, types='one_hot', drop_first=True):
    '''A function to do one-hot or label encoding
    
    input parameters
    df_in: the input data frame
    types: if 'one_hot', do one hot encoding;  if 'label', use label encode.
    return a new DF
    
    note: 
    apply Label Encoding when:
    The cate feature is ordinal (like Primary school, high school). (or too many levels)

    '''
    from sklearn.preprocessing import LabelEncoder
    
    df=df_in.copy()
    
    if types=='label' : #use label encoding
        enc = LabelEncoder()
        for col in df.columns:
            if df[col].dtypes =='object':
                df[col] = enc.fit_transform(df[col])
    elif types=='one_hot':
        val=True
        if not drop_first: val=False
        df=pd.get_dummies(df, drop_first=val)
            
    return df            
#------------------------------------
def cross_validation(model, X, y, model_name='',kfold=10):
    '''do general cross validation based on the trained model.
    model: the trained model (such as RF, kNN, ....)
    X,  y are the data for training or testing or validation.
    kfold: for giving number of K fold
    
    cross_val_score: calculate score for each CV split
    cross_validate: calculate one or more scores and timings for each CV split
    '''
    
    from sklearn.model_selection import cross_validate
    scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']

    scores =  cross_validate(model, X, y, n_jobs=-1, cv=kfold, scoring=scoring, return_train_score=False)
#   scores = cross_val_score(model, X, y, n_jobs=-1, cv=kfold, scoring='roc_auc',error_score='raise')
    for x in scores.keys():
        print("Model={} : {:15s}  mean={:.2f};  std={:.4f}".format(model_name,x, scores[x].mean(), scores[x].std() ))
    
#-------------------------------------
def create_dataset_ts(X, y, time_steps=1):
    '''A generic function to generate many lists of length (time_steps)
    The shape of the returned data is [n, m, l], where n is number of the list
    ; m is the length of the list; l is the number of features
    by using the history (n time steps from it). It works with univariate 
    (single feature) and multivariate (multiple features) Time Series data.

    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
#-----------------------------------------------
         
#=============================================================================
# ============== below are functions for NLP ============================
#=============================================================================
NLP_info='''

'''
#---------------------------------------------------
def text_vectorizer(X_train, X_test, type='tf'):
  '''
  CountVectorizer(): (BOW: bag of word) tokenize the input text data to get a matrix.
  the row is the # of documents (or messages), and the column is the unique words. 
  The value in the matrix is the TF (term frequency) for the document at the row.
  TfidfVectorizer(): BOW for term frequency and inverse document frequency. The
  matrix is scaled by IDF. TfIdf=Tf X Idf  where IDF=log((1+n)/(1+m)); 
  n is # of docs; m is # of docs with the term.

  TfidfTransformer:  scales down the values of words that occur among all texts. 
  X_train: list of text for training
  X_test: list of text for testing
  type: the type of vectorization. 
        if ='tf', use CountVectorizer(), is='tfidf' use TfidfVectorizer()
  '''
  if (type=='tf'):
    vect = CountVectorizer(decode_error='ignore')
    x_train_v = vect.fit_transform(X_train)
    x_test_v = vect.transform(X_test)
  else:
    vect = TfidfVectorizer()
    x_train_v = vect.fit_transform(X_train)
    x_test_v = vect.transform(X_test)

  return x_train_v, x_test_v, vect


#-----------------------------------------------------------------------------------------
# ============== below are functions for ploting  ============================
#-----------------------------------------------------------------------------------------
plot_info='''

'''
#---------------------------------------------------
def plot_outlier(df, types='scale', target=False, ncol=3, figsize=(), fontsize=13):
    '''A general function to plot outliers for each numerical feature.
    
    Parameters as below:
    df: a dataframe.
    types: types for ploting (scale', 'unscale', 'density')
        'scale', scale all together and plot; 
        'unscale', plot each feature by ncol; 
        'density', plot density & outliers for each class (for classification only!)
    ncol: given number of columns for plot. number of rows is auto-determined
    figsize: plot the size of the figure. if not given, auto-determined (3X5)
    target: the target for classification. If not given, only plot scale or unscale.
    fontsize: the font size for ploting.
    '''
        
    import matplotlib.gridspec as gridspec

    df_num=df.select_dtypes(include=['number'])    
    nrow=int(df_num.shape[1]/ncol) +1   #determine the num of rows 
       
    if types=='scale' :
        if len(figsize)==0: figsize=(5*ncol, 1*nrow)
        plt.figure(figsize = figsize)  
        # scale_type='Robust', 'MinMax', 'Standard'
        df0=data_scalor(df_num, df_test='', data_type='df', scale_type='standard')
        df0.boxplot(rot=0, fontsize=fontsize, figsize=figsize, vert=False )
        plt.show()
        return

    if types=='unscale' :
        if len(figsize)==0: figsize=(4*ncol, 1*nrow)
        plt.figure(figsize = figsize)  
        plotnumber = 1
        nrow=int(df_num.shape[1]/ncol)
        for col in df_num.columns:
            ax = plt.subplot(nrow +1, ncol, plotnumber)            
            sns.boxplot(df_num[col])
            plt.xlabel(col, fontsize = fontsize)
                           
            plotnumber += 1
        plt.tight_layout()
        plt.show() 
        return
    
#------  below is for to plot with different target classes

    if not target : 
        print("Wrong input. If ploting density & outliers, you need to provide target.")
        return
    if len(df[target].unique()) > 10:
        print('Warning: Nothing plotted. Only plot classification with class < 10.\n')
        return
    
    df_num['class_plot']='class_' + df_num[target].astype(str)
    print('The class values=', df_num[target].unique())
    for feature in df_num.columns:
        plt.figure(figsize = (8,4))  
        if feature == target or feature == 'class_plot' : continue
        gs1 = gridspec.GridSpec(4,1)
        ax1 = plt.subplot(gs1[:-1])
        ax2 = plt.subplot(gs1[-1])
        gs1.update(right=1.00)
        for i in df_num[target].unique():
            sns.kdeplot(df_num[feature][df_num[target]==i],ax=ax1,label=str(i))
        sns.boxplot(x=feature,y='class_plot',data=df_num, ax=ax2)

        ax2.yaxis.label.set_visible(False)
        ax1.xaxis.set_visible(False)
        plt.show()
              
#---------------------------------------------------
def plot_dist(df, ncol=3, figsize=(), types='hist', target=False, 
              features=False, fontsize=15):
    '''plot histogram/density for each numerical feature that is separated 
    with each target. The feature that separates the target well must be an 
    important one for classification!
    
    Parameters as below:
    df: the input dataframe.
    ncol: given number of columns for plot. number of rows is auto-determined
    figsize: plot the size of the figure. if not given, auto-determined (3X5)
    types: 'hist' for histogram; 'density' for density plot (scaled)
    target: the target for classification. 
    features: the features for ploting. If given, plot the features only.
    fontsize: font size for ploting

    '''
    import itertools
    
    palette = itertools.cycle(sns.color_palette())
    
    dfn=df.copy()
    if features: dfn=dfn[features]
        
    df_num=dfn.select_dtypes(include=['number'])
    df_cat=dfn.select_dtypes(exclude=['number'])
    
    if len(df_num)>0: #process numerical data
        print("\nPloting the numerical data.\n")

        columns=list(df_num.columns.values)    
        nrow=int(df_num.shape[1]/ncol) +1   #determine the num of rows    
        if len(figsize)==0: figsize1=(5*ncol, 3*nrow)
        
        fig=plt.figure(figsize=figsize1)
        for i, var_name in enumerate(columns):
            ax=fig.add_subplot(nrow, ncol, i+1)
            if (not target) :
                if types=='hist':
                    df[var_name].hist(bins=15,ax=ax)
                elif types=='density':
                    sns.distplot(df[var_name],kde=1)
            else:
                if types=='hist':
                    df.groupby(target)[var_name].hist(bins=15,ax=ax)  
                elif types=='density':
                    for j in df[target].unique():
                        sns.distplot(df[var_name][df[target]==j],kde=1,label='{}'.format(j))
                else:
                    print('\nGiven wrong option. Types should be given hist or density.\n')
                    return  
                
            plt.xlabel(var_name, fontsize = fontsize)    
            ax.set_title(var_name+"")        
        fig.tight_layout()  # Improves appearance a bit.
        plt.show()
        
    if len(df_cat)>0: #process categorical data. only plot the first 20 levels
        print("\nPloting the categorical data...\n")
        columns=list(df_cat.columns.values)    
        nrow=int(df_cat.shape[1]/ncol) +1   #determine the num of rows    
        if len(figsize)==0: figsize2=(5*ncol, 5*nrow)
        fig=plt.figure(figsize=figsize2)

        plotnumber = 1
        for col in columns:
            ax = plt.subplot(nrow +1, ncol, plotnumber)  
            if (not target) :
                df[col].value_counts().head(20).plot.bar(rot=45)
            else:
                ddf=df.groupby(target)[col].value_counts().unstack(0).head(20)
                ddf.sort_values(by=list(ddf.columns), ascending=False).plot.bar(ax=ax, rot=45)
            plt.xlabel(col, fontsize = fontsize)    
            plotnumber += 1
        
        plt.tight_layout()
        plt.show() 
        
#---------------------------------------------------
def plot_xy(df, y, xs='', types='bar', hues='', ncol=2, figsize=(), fontsize=14):
    '''A function to plot x & y combined with hue (lmplot, scatter, bar). 
    The purpose of the function is to tell how the data points separated with target.
    
    Parameters as below:
    df: an input dataframe.
    y:  must be a numerical feature name. Normally, it is the target!
    xs: input one or a list of features. if types='lmplot' or scatter, it must be 
        numerical. if types='bar', it must be categorical values
    hues: a column used for color encoding. it must be one or a list of feature names. 
          it must be categorical for any types.
    ncol: given number of columns for plot. if ploting 'bar' or 'scatter', number  
          of rows is auto-determined based on ncol. 
    types: 'bar' for histogram, 'scatter' for scatter plot, 'lmplot' for lmplot plot. 
    figsize: plot the size of the figure. if not given, auto-determined (3X5)
    fontsize: font size for ploting

    '''
    
    if len(xs)==0:
        xs=list(df.columns[df.columns!=y])
        
    
    if type(xs) is not list : xs=[xs]
    if type(hues) is not list : hues=[hues]
    
    if len(hues)==0:
        nrow=int(len(xs)/ncol) +1   #determine the num of rows 
    else:
        nrow=int(len(xs)*len(hues)/ncol) +1   
            
    if len(df[y].unique())<20 : # possible descrete number
        print('Warning: the unique value of Y < 20. Maybe not be numerical!')
        
        
    if types=='lmplot': #special for lmplot. NO ax, can not go with others 
        height=6
        aspect=1.2
        if len(figsize)>0 : height=figsize[0]
        for x in xs:
            if (df[x].dtypes !='int') and (df[x].dtypes !='float'):
                print('Wrong X is not int64 or float64. Skiping plot.')
                continue
                
            for hue in hues: 
                if len(hue)==0: hue=None
                sns.lmplot(x=x, y=y, data=df, hue=hue, palette='Set2', 
                           height=height, aspect=aspect)
                plt.show()
        return
    
    
#below is to plot the bar & scatter plot with target.
    
    if len(figsize)==0: figsize=(5*ncol, 4*nrow)
    plt.figure(figsize=figsize)
    plotnumber = 1
    for x in xs:
        for hue in hues:
            if x==hue: continue
            ax = plt.subplot(nrow, ncol, plotnumber)   
            if len(hue)==0: hue=None
            if types=='scatter':
                sns.scatterplot(data=df, x=x, y=y, hue=hue, s=15)
            elif types=='bar':
                nx_point=len(df[x].unique())
                if nx_point>50:
                    print(f'Skipped ploting. Number of point {nx_point} on {x} \
axis >50. Try scatter plot')
                    continue
                sns.barplot(data=df, x=x, y=y, hue=hue, palette='cool') 
            
            plt.xlabel(x, fontsize = fontsize)    
            plotnumber += 1
        
    plt.tight_layout()
    plt.show() 

   
#---------------------------------------------------
def plot_corr(df, figsize=(14,14),fontsize=15):
    '''A general function to plot Pearson Correlation
    
    Parameters as below:
    df: a dataframe.
    figsize: plot the size of the figure. should be adjusted accordingly.
    fontsize: font size for ploting
    features: the features for ploting. If given, plot the features only.
    '''        
  
    df_num=df.select_dtypes(include=['number'])

    colormap = plt.cm.RdBu
    plt.figure(figsize=figsize)
    plt.title('Pearson Correlation', y=1.0, size=fontsize)
    pcorr=df_num.astype(float).corr()
    sns.heatmap(pcorr,linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)

#---------------------------------------------------
def plot_target(df, target, figsize=(8,6), fontsize=15):
    '''A function to plot the target. (check balance for classification and 
    outlier/distribution for regression)
    
    Parameters as below:
    df: a dataframe.
    target: the name of the target. 
    figsize: plot the size of the figure. should be adjusted accordingly.
    fontsize: font size for ploting
    ''' 
    
    if df[target].dtypes =='float' or df[target].dtypes =='int' and len(df[target].unique())>20:
        fig, ax =plt.subplots(nrows=2,ncols=1, figsize=figsize)
        sns.boxplot(df[target], ax=ax[0])
        sns.distplot(df[target], ax=ax[1])
        plt.show()
    else: #cate       
        fig, ax =plt.subplots(nrows=1,ncols=2, figsize=figsize)
        labels=list(df[target].unique())
        sns.countplot(x=df[target], data=df, palette="pastel", ax=ax[0], edgecolor=".3")
        vc=df[target].value_counts()
        vc.plot.pie(autopct="%1.2f%%", ax=ax[1], colors=['#66a3ff','#facc99'], 
                    labels=labels, explode = (0, 0.05), startangle=120,
                    textprops={'fontsize': fontsize, 'color':'#0a0a00'})
        plt.show()

#---------------------------------------------------
def plot_target_feature_corr(df, target="class", figsize=''):
    '''A function to plot correlation between target and other features
    '''

    if len(figsize)==0: figsize=(10,5)
    plt.figure(figsize=figsize)
    df.corr()[target].sort_values(ascending = False).plot(kind='bar', rot=45)
    
    
#---------------------------------------------------
def plot_feature_importance(X, y, model=RandomForestClassifier(),nfeature=30):
    '''plot important features
    X: the training data;  y the target for training data
    '''

    model=RandomForestClassifier()
    model.fit(X,y)
#    print(model.feature_importances_)  

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat=feat_importances.nlargest(nfeature).sort_values(ascending=True)
    feat_importances.nlargest(nfeature).sort_values(ascending=True).plot(kind='barh')
    plt.show() 
    #feature=feat[feat>0.03].index
    return feat

#---------------------------------------------------
def plot_precision_and_recall(model,X_train, y_train, figsize=(10, 6)):
  # getting the probabilities of our predictions
  from sklearn.metrics import precision_recall_curve
  y_scores = model.predict_proba(X_train)[:,1]
  precision, recall, threshold = precision_recall_curve(y_train, y_scores)

  plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=3)
  plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=3)
  plt.xlabel("threshold", fontsize=12)
  plt.legend(loc="center right", fontsize=12)
  plt.ylim([0, 1])
  plt.grid()
  plt.figure(figsize=figsize)
  plt.show()
    
#------------------------------------------------------
def plot_precision_vs_recall(model,X_train, y_train, figsize=(10, 6)):
  # getting the probabilities of our predictions

  from sklearn.metrics import precision_recall_curve
  y_scores = model.predict_proba(X_train)[:,1]
  precision, recall, threshold = precision_recall_curve(y_train, y_scores)

  plt.plot(recall, precision, "g--", linewidth=2.5)
  plt.ylabel("recall", fontsize=14)
  plt.xlabel("precision", fontsize=14)
  plt.axis([0, 1.01, 0, 1.01])
  plt.grid()
  plt.figure(figsize=figsize)
  plt.show()
    
#-------------------------------------------------------------------------
def plot_deep_learning_curve(history):
    
    #plot the model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    #plot the loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
#-------------------------------------------------------------------------

def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
#-------------------------------------------------------------------------

def plot_deep_learning_curve_reg(history):
    
    #plot the model accuracy
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title("Model mean_absolute_error")
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    #plot the loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.title("Model loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
#-------------------------------------------------------------------------

def plot_roc(y_test, y_pred_prob, model_roc_auc):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC_ROC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('AUC_ROC')
    plt.show()
#-------------------------------------------------------------------------
def plot_decompose_ts(y):
  '''visulize distinct components: trend, seasonality, and noise.
  y : a series data with date as index (df['rgu'])
  '''
  from pylab import rcParams
  import statsmodels.api as sm

  rcParams['figure.figsize'] = 12, 8
  decomposition = sm.tsa.seasonal_decompose(y, model='additive')
  fig = decomposition.plot()
  plt.show()  
#plot_decompose_ts(d1['rgu'])
#-------------------------------------------------------------------------

def plot_df_ts(df):
  '''plot each column. The index should be time series
  '''
  fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,8))
  for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='blue', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

  plt.tight_layout();
#plot_df_ts(df)

#-------------------------------------------------------------------------
def plot_validation_ARIMA(model, ts, start_date, ylab=' ', start_plot=''):
  '''A function to plot and forcast the future (from last date to some step)
  model: the ARIMA model built against the data
  ts: A series with date as index
  start_date: a date from here to the end
  ylab: the label for y
  '''

  if len(start_plot)==0: start_plot=start_date
  pred = model.get_prediction(start=pd.to_datetime(start_date), dynamic=False)
  pred_ci = pred.conf_int()
  ax = ts[start_plot :].plot(label='Observed') #original
  pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, 
                           figsize=(12, 6))
  ax.fill_between(pred_ci.index,
                  pred_ci.iloc[:, 0],
                  pred_ci.iloc[:, 1], color='k', alpha=.2)
  ax.set_xlabel('Date')
  ax.set_ylabel(ylab)
  plt.legend()
  plt.show()

  y_forecasted = pred.predicted_mean
  y_truth = ts[start_date :]
  mse = ((y_forecasted - y_truth) ** 2).mean()
  rmse=round(np.sqrt(mse), 2)
  print('The Mean Squared Error (MSE) is {}. RMSE={}'.format(round(mse, 2), rmse))

  pred_ci['forecast']=pred.predicted_mean
  pred_ci['true']=y_truth
  return pred_ci

#dd=plot_validation_ARIMA(model=results, ts=d1['rgu'], start_date='2020-03-09', 
#                         ylab=' ', start_plot='2018-03-09')

#-------------------------------------------------------------------------
def plot_conf_matrix(y_true, pred, size=6):
  CM = confusion_matrix(y_true, pred)
  fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(8, 8))
  plt.show()

#plot_conf_matrix(y_true, pred, size=6)
#-------------------------------------------------------------------------

def plot_conf_matrix0(model, class_label, y_test, y_pred, size=8):
  '''
  class_label: the labels for the class.  default size is 8
  '''
  from sklearn.metrics import confusion_matrix

  fig, ax = plt.subplots(figsize=(size,size))         # Sample figsize in inches
  cm=confusion_matrix(y_test, y_pred)
  sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, 
              xticklabels=class_label,  yticklabels=class_label, ax=ax)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.show()
  
#-------------------------------------------------------------------------
def plot_wordcloud(text, fig_size=(18,10), title=''):
  '''A function plot_wordcloud(text, fig_size=(18,10), title='')
  to plot WordCloud.  WordCloud representation of most used words in each doc.
  '''
  from wordcloud import WordCloud,STOPWORDS

  plt.figure(figsize=fig_size)

  cloud_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

#plt.subplot(2, 3, 1)
  plt.axis('off')
  plt.title(title,fontsize=40)
  plt.imshow(cloud_toxic)

#-------------------------------------------------------------------------
def plot_tree(model, max_depth=3, feature_name=[], class_name=''):
    '''
    model: should be the decisen tree model.
    '''

    from sklearn import tree
    import graphviz
    
# DOT data  'test-tree.png'
    dot_data = tree.export_graphviz(model, max_depth=max_depth, label='all', filled=True,
                                    feature_names=feature_name, class_names=class_name )

# Draw graph
    graph = graphviz.Source(dot_data, format="png") 
    return graph

#-------------------------------------------------------------------------
#========================================================================================
def optimize_SARIMAX(ts, num):
  ''' A function to use a grid search to find the optimal set of parameters  
  that yields the best performance for our model.
  ts: A series with date as index (work when: d1=d1.resample('W').mean() )
  num: a given number for various orders

  After running, check the DF and use the order to the model
  '''

  import itertools
  from statsmodels.tsa.arima_model import ARIMA
  from statsmodels.tsa.statespace.sarimax import SARIMAX

  p = d = q = range(0, num)
  pdq = list(itertools.product(p, d, q))  #combinations for Seasonal ARIMA.
  seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

  par=[]
  for param in pdq:
    for param_seasonal in seasonal_pdq:
      try:
        mod = SARIMAX(ts,
                      order=param,
                      seasonal_order=param_seasonal,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        results = mod.fit()
        par.append([param, param_seasonal, results.aic])
      except:
#        print('Error occurred')
        continue
  df_par=pd.DataFrame(data=par, columns=['param', 'param_seasonal', 'results.aic'])
  return df_par.sort_values(by=['results.aic'])

#dd= optimize_SARIMAX(d1['rgu'], num=2) 

#-------------------------------------------------------------------------
def forecast_future_ARIMA(model, ts, steps, ylab=' ', plot=False):
  '''A function to plot and forcast the future (from last date to some step)
  model: the ARIMA model built against the data
  ts: A series with date as index
  steps: a given number of steps forward (e.g 10 weeks, 10 month)
  ylab: the label for y
  '''
  pred_uc = model.get_forecast(steps)
  pred_ci = pred_uc.conf_int()  # a DF
  pred_mean=pred_uc.predicted_mean
  pred_ci['avg']=pred_mean.tolist()
  if not plot: return pred_ci

  ax = ts.plot(label='Observed', figsize=(12, 6))
  pred_mean.plot(ax=ax, label='Forecast')
  ax.fill_between(pred_ci.index,
                  pred_ci.iloc[:, 0],
                  pred_ci.iloc[:, 1], color='k', alpha=.25)
  ax.set_xlabel('Date')
  ax.set_ylabel(ylab)
  plt.legend()
  plt.show()
  return pred_ci

#dd=forecast_future_ARIMA(model=results, ylab='RGU', ts=d1['rgu'], steps=20, plot=True)

#---------------------------------------------------
def arima_forcast(df, target, train_size):
  '''A function to use ARIMA for 1D forcasting
  Autoregressive Integrated Moving Average (ARIMA)
  It combines both Autoregression (AR) and Moving Average (MA) models as well 
  as a differencing pre-processing step of the sequence to make the sequence 
  stationary, called integration (I).
  order=(3,1,1)  AR(p), I(d), and MA(q)

  df: a dataframe containing target, timestamp as index
  target: a column for prediction
  train_size: a percentage of df used for training
  '''

  from statsmodels.tsa.arima_model import ARIMA
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from sklearn.metrics import mean_squared_error
  import math

  X = df[target]
  size = int(len(X) * train_size)
  print('train size=', size)
  train, test = X[0:size], X[size:len(X)]
  history = [x for x in train]
  predictions = list()
  #predicting one at a time
  for i in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
#    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[i]
    history.append(obs) #adding one value from test to train
#    print(t+1, 'predicted=%f, expected=%f' % (yhat, obs))
   
  error = mean_squared_error(test, predictions)
  print('Test RMSE: %.2f' % math.sqrt(error))

  dd=pd.DataFrame(data=test)
  if type(predictions[0])==list:
    dd['pred']=predictions
  else:
    dd['pred']=[x.tolist()[0] for x in predictions]
  dd.plot(figsize=(10,5))
  train.plot(figsize=(10,5))
  plt.show()
  return dd
#dd=arima_forcast(d1, target='rgu', train_size=0.95) 

#--------------------------------------------------------
def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    fscore = f1_score(y_actual,(y_pred > thresh) )
    specificity = sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('fscore:%.3f'%fscore)
    print('specificity:%.3f'%specificity)
    print(' ')
    return auc, accuracy, recall, precision,fscore, specificity 
#--------------------------------------------------------

def logit_prob(reg, x_test):
    '''logit_prob(reg, x_test): calc prob from x_test and return prob.
    '''
    import numpy as np
    logit=x_test.dot(reg.coef_.T) + reg.intercept_
    p=1./(1+np.exp(-logit))
    return p

#-----------------------------
import sys
import warnings
import math
import statsmodels
import numpy as np
from scipy import stats
import statsmodels.api as smf

def firth_likelihood(beta, logit):
    return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))

#-----------------------------
# Do firth regression
# Note information = -hessian, for some reason available but not implemented in statsmodels
def fit_firth(y, X, start_vec=None, step_limit=1000, convergence_limit=0.0001):

    logit_model = smf.Logit(y, X)
    
    if start_vec is None:
        start_vec = np.zeros(X.shape[1])
    
    beta_iterations = []
    beta_iterations.append(start_vec)
    for i in range(0, step_limit):
        pi = logit_model.predict(beta_iterations[i])
        W = np.diagflat(np.multiply(pi, 1-pi))
        var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))

        # build hat matrix
        rootW = np.sqrt(W)
        H = np.dot(np.transpose(X), np.transpose(rootW))
        H = np.matmul(var_covar_mat, H)
        H = np.matmul(np.dot(rootW, X), H)

        # penalised score
        U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))
        new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

        # step halving
        j = 0
        while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
            new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
            j = j + 1
            if (j > step_limit):
                sys.stderr.write('Firth regression failed\n')
                return None

        beta_iterations.append(new_beta)
        if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
            break

    return_fit = None
    if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
        sys.stderr.write('Firth regression failed\n')
    else:
        # Calculate stats
        fitll = -firth_likelihood(beta_iterations[-1], logit_model)
        intercept = beta_iterations[-1][0]
        beta = beta_iterations[-1][1:].tolist()
        bse = np.sqrt(np.diagonal(np.linalg.pinv(-logit_model.hessian(beta_iterations[-1]))))
        
        return_fit = intercept, beta, bse, fitll

    return return_fit

#==================================================================
def logit_Firth(X, y):
    ''' logit_firth(X, y)
    Python implementation of Firth regression by John Lees
    See https://www.ncbi.nlm.nih.gov/pubmed/12758140
    '''

  
    # create X and y here. Make sure X has an intercept term (column of ones)
    # ...

    # How to call and calculate p-values
    (intercept, beta, bse, fitll) = fit_firth(y, X)
    beta = [intercept] + beta
    
    # Wald test
    waldp = []
    for beta_val, bse_val in zip(beta, bse):
        waldp.append(2 * (1 - stats.norm.cdf(abs(beta_val/bse_val))))

    # LRT
    lrtp = []
    for beta_idx, (beta_val, bse_val) in enumerate(zip(beta, bse)):
        null_X = np.delete(np.array(X), beta_idx, axis=1)
        (null_intercept, null_beta, null_bse, null_fitll) = fit_firth(y, null_X)
        lrstat = -2*(null_fitll - fitll)
        lrt_pvalue = 1
        if lrstat > 0: # non-convergence
            lrt_pvalue = stats.chi2.sf(lrstat, 1)
        lrtp.append(lrt_pvalue)
    
    df_result=pd.DataFrame({'features':X.columns.tolist(),'Beta':beta, 'Pvalue':lrtp, 'Wald_test':waldp })
    return(df_result, intercept)

# ------------------------------------------------------------------------------
def get_df_pca(df_in, loading=0.9, figsize=(8,6) ):

    from sklearn.decomposition import PCA
    #plot PCA
    df=df_in.copy()
    pca=PCA().fit(df)
    ratio=pca.explained_variance_ratio_
    plt.figure(figsize=figsize)
    plt.plot(range(0, len(ratio)), ratio.cumsum(), marker='o', linestyle='--')
    plt.grid()
    plt.show()

#get number of components if ask to explain variance of 0.95 
    npca=PCA(loading).fit(df).n_components_
    print('Number of components {:2d} corresponding to the loading factor of {:.2f}.'.format(npca,loading))

#final PCA and check clusters
    pca_f=PCA(n_components = npca).fit(df)
    df_tran=pca_f.transform(df)
    return df_tran

# ------------------------------------------------------------------------------
def groupby_target(df, col_inp=[], target=['target']):
    '''
    col_inp: the given columns for agg.
    target: the target or a list of INT variables
    '''
#    dfn=df.select_dtypes(include=['int64'])
    
    dfn=df.copy()
    print('Aggregating {} for the given (or all) features.'.format(target))
#    dfm=pd.DataFrame(columns=['feature','count', 'mean', 'sum', 'std', 'max', 'min'])
#  if specify columns, the DF will use it, but multiple index may be affected!!
    dfm=pd.DataFrame() 
    col=col_inp
    if len(col_inp) ==0 : col=dfn.columns.tolist()
    y=target
    for x in col:
        dft=dfn.groupby(y)[x].agg(['count', 'mean', 'sum', 'std', 'max', 'min'])
        dft['feature'] = x
        
        dfm=dfm.append(dft)
        
    dfm1= dfm.reset_index().sort_values(['feature']+target)
    dfm2= dfm1.reset_index(drop=True)
    return dfm2
        
#ddm1=groupby_target(df1, col_inp=[], target=['Participant', 'CKD_Flag'])