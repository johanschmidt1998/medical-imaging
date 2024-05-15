import pandas as pd
import numpy as np
import itertools
import csv

def get_rating_scale_from_df(df):
    # returns the range of ratings, assuming all ratings are present.
    return (np.unique(df)[~np.isnan(np.unique(df))])



def get_column_total_from_df(df):
    return df.count()

def get_empty_matrix(scale):
    return np.zeros((scale,scale))

def count_pairs(col_tot):
    return col_tot * (col_tot-1)

def update_coincidence_matrix(k_pairs, matrix, mu):
    for i in k_pairs:
        c,k = int(i[0])-1,int(i[1])-1
        matrix[c][k] += 1/(mu-1)
    return matrix

def make_coincidence_matrix(df):

    scale_len = len(get_rating_scale_from_df(df))

    coincidence_matrix = np.zeros((scale_len,scale_len))

    col_tot = get_column_total_from_df(df)
    
    n_columns = df.shape[1]

    for k in range(n_columns): 
        mu = col_tot[k]

        k_entries = np.array(df.iloc[:,k])
        k_entries = k_entries[~np.isnan(k_entries)] 
        k_pairs = list(itertools.permutations(k_entries,2))
        
        coincidence_matrix = update_coincidence_matrix(k_pairs, coincidence_matrix, mu)

    return coincidence_matrix

def get_difference_matrix(coincidence_matrix, rating = None, method = "nominal"): # uses the previously created coincidence matrix
    methods = ["nominal", "interval", "ordinal", "ratio", "bipolar"]
    if method not in methods:
        print(f'Error: Incorrect method. Please set method to one of the following: {methods}')
        pass
    
    if method == "bipolar":
        if rating.all() == None:
            print(f'Please include the rating scale, as a list (e.g. [-1,0,1]). \n You can use get_rating_scale_from_df(<df>)')
            pass
        cmin,cmax = np.min(rating), np.max(rating)

    nc,nk = coincidence_matrix.shape[0], coincidence_matrix.shape[1]
    difference_matrix = np.zeros((nc,nk))

    for c in range(nc):
        for k in range(nk):

            if method == "nominal":
                difference_matrix = np.ones((nc,nk))
                return np.triu(difference_matrix,1) + np.triu(difference_matrix,1).T
            
            elif method == "interval":
                difference_matrix[c][k] = (c-k)**2

            elif method == "ordinal":
                if c <=k:
                    sumc = (sum(coincidence_matrix[c]))
                    sumk = (sum(coincidence_matrix[:][k]))
                    n = ((sumc+sumk)/2)
                
                    ord = ((np.sum(coincidence_matrix[c:k+1])))
                    ord_squared = (ord - n)**2
                    difference_matrix[c][k] = ord_squared
            
            elif method == "ratio":
                if (c-k) == (c+k) == 0:
                    difference_matrix[c][k] = 0
                else:
                    difference_matrix[c][k] = ( (c-k)/(c+k) )**2

            elif method == "bipolar":
                difference_matrix[c][k] = ((rating[c]-rating[k])**2) / ( (rating[c] + rating[k] - 2*cmin) * (2*cmax-rating[c]-rating[k]) )

    return np.triu(difference_matrix,1) + np.triu(difference_matrix,1).T

def get_nuber_of_values(coincidence_matrix):
    return np.sum(coincidence_matrix)

def krippendorff_alpha(df,rating = None, method="nominal"):
    top,bot = 0,0

    coincidence_matrix = make_coincidence_matrix(df)
    
    n = get_nuber_of_values(coincidence_matrix)

    difference_matrix = get_difference_matrix(coincidence_matrix,rating = rating, method = method)

    m = range(len(coincidence_matrix))

    for c in m:
        for k in m:
            if k > c:
                top += coincidence_matrix[c][k] * difference_matrix[c][k] 
                bot += sum(coincidence_matrix[c]) * sum(coincidence_matrix[:][k]) * difference_matrix[c][k]
            else:
                continue

    alpha = 1 - (n-1) * (top/bot)

    return (f'\n Krippendorff {method} alpha: {round(alpha,3)}')

df = pd.read_csv("C:\\Users\\elias\\Desktop\\Projects in data science\\annotators_guide_proper_format.csv")
annotator1_list = list(df['Annotator 1'])
annotator2_list = list(df[ 'Annotator 2'])
annotator_1and2 = {
        'anno_1': annotator1_list,
        'anno_2': annotator2_list} 

df = pd.DataFrame.from_dict(annotator_1and2, orient='index')
print(krippendorff_alpha(df,rating = None, method="ordinal"))