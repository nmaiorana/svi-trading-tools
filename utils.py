import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20, 8)
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
import numpy as np

def plot_results(tree_sizes, train_score, oob_score, valid_score, legend, title, xlabel):
    plt.plot(tree_sizes, train_score, 'xb-');
    plt.plot(tree_sizes, oob_score, 'xg-');
    plt.plot(tree_sizes, valid_score, 'xr-');
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel('Accuracy')
    plt.legend(legend)
    #plt.ylim(y_range[0], y_range[1]);
    
    plt.show()
    
    
def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))

def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    ranked_features = []
    for x_train_i in range(len(importances)):
        ranked_features.append(feature_names[indices[x_train_i]])
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))
        
    return ranked_features
        
def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Generate the train, validation, and test dataset.

    Parameters
    ----------
    all_x : DataFrame
        All the input samples
    all_y : Pandas Series
        All the target values
    train_size : float
        The proportion of the data used for the training dataset
    valid_size : float
        The proportion of the data used for the validation dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_valid : DataFrame
        The validation input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_valid : Pandas Series
        The validation target values
    y_test : Pandas Series
        The test target values
    """
    assert train_size >= 0 and train_size <= 1.0
    assert valid_size >= 0 and valid_size <= 1.0
    assert test_size >= 0 and test_size <= 1.0
    assert train_size + valid_size + test_size == 1.0
    
    all_dates = all_x.index.levels[0]
    train_end = int(all_dates.shape[0]*(train_size)) - 1
    valid_end = train_end + int(all_dates.shape[0]*valid_size)
    
    X_train = all_x.loc[:all_dates[train_end],]
    X_valid = all_x.loc[all_dates[train_end + 1]:all_dates[valid_end],]
    X_test  = all_x.loc[all_dates[valid_end + 1]:]
    
    y_train = all_y.loc[:all_dates[train_end],]
    y_valid = all_y.loc[all_dates[train_end + 1]:all_dates[valid_end],]
    y_test  = all_y.loc[all_dates[valid_end + 1]:]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Get the non overlapping samples.

    Parameters
    ----------
    x : DataFrame
        The input samples
    y : Pandas Series
        The target values
    n_skip_samples : int
        The number of samples to skip
    start_i : int
        The starting index to use for the data
    
    Returns
    -------
    non_overlapping_x : 2 dimensional Ndarray
        The non overlapping input samples
    non_overlapping_y : 1 dimensional Ndarray
        The non overlapping target values
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    
    dates = sorted(list(set(x.index.get_level_values(0))))
    non_overlapping_dates = dates[start_i::n_skip_samples+1]
    x_sub =  x.loc[non_overlapping_dates]
    y_sub = y.loc[non_overlapping_dates]
    
    return x_sub, y_sub