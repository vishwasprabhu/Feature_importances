import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import inspect
import matplotlib.pyplot as plt
from sklearn.base import clone
import xgboost
import shap
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

def SpearmanCoeff(df,col1, col2):
    rank1 = df[col1].rank()
    rank2 = df[col2].rank()
    rank = 1-6*np.sum((rank1-rank2)**2)/(df.shape[0]**3-df.shape[0])
    return rank


def Spearman_imp(df, target, probs=False):
    X_data = df.loc[:, df.columns != target]
    ss = StandardScaler()
    X = ss.fit_transform(X_data)
    y = df.loc[:, target].values
    sc = []
    for i in range(X.shape[1]):
        sc.append(SpearmanCoeff(df, X_data.columns[i], target))
    if probs:
        return len(sc) - rankdata(sc).astype(int), sc
    return len(sc) - rankdata(sc).astype(int)

def mrmr_imp(df, target, probs=False):
    X = df.loc[:,df.columns!=target]
    S = []
    sc = []
    rank=[]
    J = {}
    element_left = set(X.columns)
    for i in range(len(X.columns)):
        J = mrmr_(df, target, S)
        S.append(np.where(X.columns==max(J))[0][0])
        sc.append(J[max(J)])
    feat_imp = [x for _, x in sorted(zip(S, sc))]
    if probs:
        return len(feat_imp) - rankdata(feat_imp).astype(int), feat_imp
    return len(feat_imp) - rankdata(feat_imp).astype(int)


def mrmr_(df, target, S = [2]):
    X = df.loc[:,df.columns!=target]
    J = {}
    for ele in (set(list(X.columns)) - set(list(X.columns[S]))):
        Ixy = abs(SpearmanCoeff(df,ele,target))
        Ixx = 0
        for j,col in enumerate(X.columns):
            if ele != col and j in S:
                Ixx += abs(SpearmanCoeff(X,ele, col))
        if len(S)>0:
            J[ele] = Ixy - Ixx/len(S)
        else:
            J[ele] = Ixy
    return J


def pca_imp(df,target, probs = False):
    X = df.loc[:,df.columns!=target]
    ss = StandardScaler()
    X = ss.fit_transform(X)
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    sc = abs(pca.components_[0])
    if probs:
        return len(sc) - rankdata(sc).astype(int), sc
    return len(sc) - rankdata(sc).astype(int)


def ols_imp(df,target, probs=False):
    X_data = df.loc[:,df.columns!=target]
    ss = StandardScaler()
    X = ss.fit_transform(X_data)
    y = df.loc[:, target].values
    model = sm.OLS(y,X)
    results = model.fit()
    sc = abs(results.params)
    if probs:
        return len(sc) - rankdata(sc).astype(int), sc
    return len(sc) - rankdata(sc).astype(int)


def dropcol_importances(model,df,target):
    X = df.loc[:,df.columns!=target]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    baseline = mean_absolute_error(y_val, model.predict(X_val))
    imp = []
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_val_ = X_val.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X_train_, y_train)
        m = mean_absolute_error(y_val, model_.predict(X_val_))
        imp.append(m - baseline)
    return len(imp) - rankdata(imp).astype(int), imp


def permutation_importances(model, df, target):
    X = df.loc[:,df.columns!=target]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    baseline = mean_absolute_error(y_val, model.predict(X_val))
    imp = []
    for col in X_val.columns:
        save = X_val[col].copy()
        X_val.loc[:,col] = np.random.permutation(X_val[col])
        m = mean_absolute_error(y_val, model.predict(X_val))
        X_val.loc[:,col] = save
        imp.append(m - baseline)
    return len(imp) - rankdata(imp).astype(int), imp

def plot_feat_imp(df):
    ordered_df = df.sort_values(by='Feature_importance',ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8,6))
    my_range = range(1, len(df.index) + 1)

    ax.hlines(y=my_range, xmin=0, xmax=ordered_df['Feature_importance'],
               color='skyblue')
    for i,val in enumerate(ordered_df.Feature_importance):
        if np.sign(val)==1:
            ax.annotate(f"{val:0.3f}", (val + 0.02, i+1+0.1), size=11, annotation_clip=False, color='black')
        else:
            ax.annotate(f"{val:0.3f}", (val - 0.12, i+1+0.1), size=11, annotation_clip=False, color='black')
    ax.scatter(ordered_df['Feature_importance'], my_range, marker = 'o')
    ax.set_yticks(ordered_df.index+1)
    ax.set_yticklabels(labels = ordered_df.Feature.values,fontsize=12)
    ax.set_xlabel('Feature Importance',fontsize=12)
    if np.sign(min(ordered_df.Feature_importance))==1:
        ax.set_xlim(min(ordered_df.Feature_importance)-min(ordered_df.Feature_importance),max(ordered_df.Feature_importance)+max(ordered_df.Feature_importance)*0.5)
    else:
        ax.set_xlim(min(ordered_df.Feature_importance)+min(ordered_df.Feature_importance),max(ordered_df.Feature_importance)+max(ordered_df.Feature_importance)*0.5)
    return plt

def compare(df,target,func_ls):
    dicti = {}
    for func in func_ls:
        func_str = inspect.getsource(func).replace('(','\n').replace(' ','\n').split('\n')[1]
        dicti[func_str] = func(df, target)
    return dicti

def RF_strategy(df,target,J):
    X = df.loc[:,df.columns!=target]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    mae_dict = {}
    for func, val in J.items():
        X_train_cols = []
        mae = []
        for i in val:
            X_train_cols.append(i)
            model = RandomForestRegressor(n_estimators=10,max_depth=10,min_samples_leaf=50)
            model.fit(X_train.iloc[:,X_train_cols],y_train)
            y_pred = model.predict(X_val.iloc[:,X_train_cols])
            mae.append(mean_absolute_error(y_val, y_pred))
        mae_dict[func] = mae
    df_imp = pd.DataFrame(mae_dict)
    return df_imp

def OLS_strategy(df,target,J):
    X = df.loc[:,df.columns!=target]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    mae_dict = {}
    for func, val in J.items():
        X_train_cols = []
        mae = []
        for i in val:
            X_train_cols.append(i)
            model = sm.OLS(y_train,X_train.iloc[:,X_train_cols])
            results = model.fit()
            y_pred = results.predict(X_val.iloc[:,X_train_cols])
            mae.append(mean_absolute_error(y_val, y_pred))
        mae_dict[func] = mae
    df_imp = pd.DataFrame(mae_dict)
    return df_imp

def xgboost_strategy(df,target,J):
    X = df.loc[:,df.columns!=target]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    mae_dict = {}
    for func, val in J.items():
        X_train_cols = []
        mae = []
        for i in val:
            X_train_cols.append(i)
            model = xgboost.XGBRegressor().fit(X_train.iloc[:,X_train_cols], y_train)
            y_pred = model.predict(X_val.iloc[:,X_train_cols])
            mae.append(mean_absolute_error(y_val, y_pred))
        mae_dict[func] = mae
    df_imp = pd.DataFrame(mae_dict)
    return df_imp


def plot_strategy(df_imp):
    x = df_imp.index

    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111)

    ax.plot(x,df_imp['Spearman_imp'],c='b',marker="^",ls='--',label='Spearman_imp',fillstyle='none')
    ax.plot(x,df_imp['pca_imp'],c='g',marker=(8,2,0),ls='--',label='pca_imp')
    ax.plot(x,df_imp['ols_imp'],c='k',ls='-',label='ols_imp')
    ax.plot(x,df_imp['mrmr_imp'],c='r',marker="v",ls='-',label='mrmr_imp')

    plt.xlabel('Top k features')
    plt.ylabel('MAE error')
    plt.legend()
    return plt

def plot_strategy_shap(df_imp):
    x = df_imp.index

    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111)

    ax.plot(x,df_imp['Spearman_imp'],c='b',marker="^",ls='--',label='Spearman_imp',fillstyle='none')
    ax.plot(x,df_imp['pca_imp'],c='g',marker=(8,2,0),ls='--',label='pca_imp')
    ax.plot(x,df_imp['ols_imp'],c='k',ls='-',label='ols_imp')
    ax.plot(x,df_imp['mrmr_imp'],c='r',marker="v",ls='-',label='mrmr_imp')
    ax.plot(x,df_imp['shap'], c='b', marker="s", ls=':', label='shap_imp')

    plt.xlabel('Top k features')
    plt.ylabel('MAE error')
    plt.legend()
    return plt

def list_to_pd(df,target,ls,sc):
    return pd.DataFrame({'Feature':df.loc[:,df.columns!=target].columns,'Rank':ls,'Feature_importance':sc})

def feature_select(dataset, df, model, target):
    col = []
    val_mae = []
    X = dataset.loc[:,dataset.columns!=target]
    y = dataset[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    val_mae.append(mean_absolute_error(y_val, model.predict(X_val)))
    for i in reversed(range(1,df.shape[0])):
        col.append(df[df.Rank == i]['Feature'].values[0])
        X_train_ = X_train[list(set(df.Feature)-set(col))]
        X_val_ = X_val[list(set(df.Feature)-set(col))]
        model_ = clone(model)
        model_.fit(X_train_, y_train)
        mae = mean_absolute_error(y_val, model_.predict(X_val_))
        val_mae.append(mae)
        if mae == min(val_mae):
            j = i
    return val_mae, j


def feature_selection_plot(df, j):
    fig=plt.figure(figsize=(10,6))
    ax=fig.add_subplot(111)
    ax.plot(df['Feature'],df['val_mae'],c='b',marker="^",ls='--',fillstyle='none')
    ax.scatter(df.iloc[df.shape[0]-j,0],df.iloc[df.shape[0]-j,3],c='r',marker = 'o',edgecolors='face',s=200)
    plt.xticks(rotation=45)
    ax.set_title('Feature selection plot')
    ax.set_xlabel('Features')
    ax.set_ylabel('Validation MAE')
    val = df.iloc[df.shape[0]-j,3]
    ax.annotate(f"Optimal point MAE:{val:0.0f}", (df.shape[0]-j -1, val+1100), size=11,weight='bold', annotation_clip=False, color='black')
    return plt


def feature_imp_plot(df, dataset, frac=0.5, alpha=0.95):
    conf_dict = {}
    for col in df.Feature:
        conf_dict[col] = []
    for i in range(5):
        df_conf = dataset.sample(frac=0.5)
        ls, sc = ols_imp(df_conf, 'median_house_value', probs=True)
        df = list_to_pd(dataset, 'median_house_value', ls=ls, sc=sc)
        for col in df.Feature:
            conf_dict[col].append(df[df['Feature'] == col]['Feature_importance'].values[0])
    conf_dict_int = {}
    for col in df.Feature:
        conf_dict_int[col] = st.t.interval(alpha=0.95, df=len(conf_dict[col]) - 1, loc=np.mean(conf_dict[col]),
                                           scale=st.sem(conf_dict[col]))
    ordered_df = df.sort_values(by='Feature_importance', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    my_range = range(1, len(df.index) + 1)
    ax.hlines(y=my_range, xmin=0, xmax=ordered_df['Feature_importance'],
              color='skyblue')
    for i, col in enumerate(ordered_df.Feature):
        ax.hlines(y=i + 1, xmin=conf_dict_int[col][0], xmax=conf_dict_int[col][1], color='black')
        ax.scatter(y=[i + 1, i + 1], x=conf_dict_int[col], marker='|', color='black')

    ax.scatter(ordered_df['Feature_importance'], my_range, marker='o', color='b')
    for i, val in enumerate(ordered_df.Feature_importance):
        if np.sign(val) == 1:
            ax.annotate(f"{val:0.0f}", (val + 0.02, i + 1 + 0.1), size=11, annotation_clip=False, color='black')
        else:
            ax.annotate(f"{val:0.0f}", (val - 0.12, i + 1 + 0.1), size=11, annotation_clip=False, color='black')

    ax.set_yticks(ordered_df.index + 1)
    ax.set_yticklabels(labels=ordered_df.Feature.values, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=10)
    return plt

def feature_p_values(data,target, imp=permutation_importances,iters= 80):
    dataset = data.copy()
    model = RandomForestRegressor(n_estimators=10,max_depth=10,min_samples_leaf=50)
    feat_imp_act = imp(model,dataset,target=target)
    feat_imps = []
    for i in range(iters):
        dataset = data.copy()
        dataset['median_house_value'] = np.random.permutation(dataset.median_house_value.values)
        model = RandomForestRegressor(n_estimators=10,max_depth=10,min_samples_leaf=50)
        feat_imp = imp(model, dataset, 'median_house_value')
        feat_imps.append(feat_imp)
    return feat_imps,feat_imp_act


def p_value_plot(dataset,target):
    feat_imps, feat_imp_act = feature_p_values(data=dataset,target=target)
    X = dataset.loc[:,dataset.columns!=target]
    df_feat = pd.DataFrame(feat_imps,columns=X.columns)
    fig, ax = plt.subplots(nrows=df_feat.shape[1]//2, ncols=2,figsize=(16,16))
    fig.tight_layout(pad=5.0)
    p_value = np.sum(df_feat > feat_imp_act,axis=0)/df_feat.shape[0]
    for i in range(df_feat.shape[1]//2):
        for j in range(2):
            ax[i,j].hist(df_feat.iloc[:,2*i+j],label='Null')
            ax[i,j].vlines(feat_imp_act[2*i+j],ymin=0,ymax=10,color='r', label = 'real target')
            ax[i,j].legend(loc="upper right")
            ax[i,j].set_xlabel(f'{df_feat.columns[2*i+j]}')
            ax[i,j].set_ylabel('Repetetions')
            ax[i,j].annotate(f"p-value:{p_value[2*i+j]:0.3f}", (feat_imp_act[2*i+j], 11), size=10, annotation_clip=False, color='black')
    return plt
