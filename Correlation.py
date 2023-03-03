import matplotlib.pyplot as plt
import numpy as np


def correlation_table(df, method='pearson'): # correlation between all parameters
    '''    
    Pearson's correlation is a measure of the linear relationship between two continuous random 
    variables. It does not assume normality although it does assume finite variances and finite 
    covariance. When the variables are bivariate normal, Pearson's correlation provides a complete 
    description of the association.

    Spearman's correlation applies to ranks and so provides a measure of a monotonic relationship 
    between two continuous random variables. It is also useful with ordinal data and is robust to 
    outliers (unlike Pearson's correlation).

    The distribution of either correlation coefficient will depend on the underlying distribution, 
    although both are asymptotically normal because of the central limit theorem.

    Kendall rank correlation: Kendall rank correlation is a non-parametric test that measures the 
    strength of dependence between two variables.  If we consider two samples, a and b, where each 
    sample size is n, we know that the total number of pairings with a b is n(n-1)/2.  The following 
    formula is used to calculate the value of Kendall rank correlation:
    '''
    
    df_corr = df.corr(method=method)
    
    data1 = df_corr.values
    data1 = np.around(data1, decimals=2)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    
    for i in range(len(data1)):
        for j in range(len(data1)):
            text = ax1.text(j, i, data1[i, j], 
                            horizontalalignment='left', 
                            verticalalignment='top', color="k")              

    
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    #plt.show()
