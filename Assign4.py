import pandas as pd
import numpy as np
import numpy.linalg as LA
import random
from sklearn.preprocessing import StandardScaler 
import csv
import sys

#convert point_view to col_view given headers
def col_view(point_view, headers):
    column_view = {key: None for key in headers}
    point_view_T = np.transpose(point_view)
    column_view_iterator = 0
    for key in list(headers):
        column_view[key] = point_view_T[column_view_iterator]
        column_view_iterator += 1

    return column_view

#convert col_view to point_view
def p_view(col_view):
    col_view_list = []
    for data in col_view.values():
        d = np.array(data)
        col_view_list.append(d)

    col_view_list = np.array(col_view_list)

    return np.transpose(col_view_list)

#subset of point view given lower bound and upper bound
#preconditions: upperbound < len(point_view), lower_bound >= 0
def p_range(point_view, lower_bound, upper_bound):
    p_subset = []
    for i in range(lower_bound, upper_bound+1):
      p_subset.append(point_view[i])

    return np.array(p_subset)

        



def parse_dataset(filename):
    
    
    with open(filename) as csv_file:
        headers = next(csv.reader(csv_file))
        headers.pop(28)
        headers.pop(0)
        pd_data_frame = pd.read_csv(filename, usecols=headers)
        point_view = pd_data_frame.to_numpy()
        scaler = StandardScaler()
        scaled_pv = scaler.fit_transform(point_view)

        pv_copy = scaled_pv.copy()
        #need a way to access attributes
        column_view = col_view(pv_copy, headers)

        #split into response and independent variables
        
        #response
        response_header = headers[0]
        response_column = column_view[response_header]
        column_view.pop(response_header)
        iv_headers = headers.copy()
        iv_headers.pop(0)


        #column_view = indepenent_variables
        #make indepent_variable version in point_view form
        #reassign point_view to independent version
        point_view = p_view(column_view)
        
        

    
    return (column_view, point_view, response_header, response_column, iv_headers)
    

def augement(point_view):
    col_view = np.transpose(point_view)
    col_view_ones = np.ones(len(col_view[0]))
    col_view = col_view.tolist()
    col_view.insert(0, col_view_ones.tolist())
    col_view = np.array(col_view)
    augemented_point_view = np.transpose(col_view)

    return augemented_point_view

def ridge_regression_sgd(D, Y, alpha, eta, eps, max_iter):
    aug_D = augement(D)
    t = 0
    aug_w = np.ones(len(aug_D[0]))
    converged = False
    prev_w = aug_w
    while not converged or t == max_iter:
        for k in range(1, len(aug_D)):
            r_k = random.randrange(1, len(aug_D))
            p = aug_D[r_k]
            gradient = -((Y[r_k] - np.dot(np.transpose(p), aug_w)) * (p)) + ((alpha/len(aug_D)) * aug_w)

            aug_new_w = np.subtract(aug_w, (eta*gradient))
            prev_w = aug_w
            aug_w = aug_new_w
        t = t + 1
        print(aug_w)
        converged = LA.norm(aug_w - prev_w) <= eps
    print(t)

    return aug_w


def SSE(aug_w, validation_data):
    LA.norm()
    


if __name__ == "__main__":
    arguments = sys.argv

    filename = arguments[1]
    alpha = float(arguments[2])
    eta = float(arguments[3])
    eps = float(arguments[4])
    max_iter = float(arguments[5])

    #step 1
    #use aep data set.
    #ignore first attribute (date time variable)
    #remove last attribute
    #use first attribute (after removing date time) as the response variables, with the remaining as predictor variables
    #use standard scalar to normalize each attribute to have mean zero and variance one.
        #use the skearn StandardScalar to do this OR use formula (xi - ux) / sigx
    
    col_view, point_view, resp_header, resp_col, iv_headers = parse_dataset(filename)

    
    #point_views
    training_data = p_range(point_view, 0, 13734)
    validation_data = p_range(point_view, 13735, 13734+2000)
    testing_data = p_range(point_view, 13735+2000, 13734+2000+4000)


    learned_w = ridge_regression_sgd(training_data, resp_col, alpha, eta, eps, max_iter)
    


    #what do we have?

    #augmented training data in point and column view


    #what do we need for linear regression with regularization?
    #ridge regression inputs: points, response_column, nu, E
    #n = len(points)



    #step 2 (linear regression with Regularization)
    #implement ridge regression algorithm using batch gradient descent to solve for w.
    #Use equation 23.35 to compute the gradient at each step
    #choose appropriate step size value n and regularization constant, alpha.

    #should use the first 13735 points as training the next 2000 points as validation, and the last 4000 points for testing.
    #the validation set will be used to find the best alpha and n values.
        #for each value of alpha, first learn w on the training set then compute the SSE value on the validaton set.
        #the value that gives the lead validation SSE is the one to choose.


    #once the best alpha and w have been found, you should evaluate the model on the testing data.
    #in particular you should compute the SSE value for the predicitons on the test data. You should
    #also report the R^2 statistic on the test data which is defined as R^2 = (TSS - SSE) / TSS where TSS is the total scatter of the response variable
        #TSS = sum from 1 to n of (yi - muY)^2
    
    #for initalizing the W vector, you may use the np.ones(), a vector of all ones.

    #also very that the w vector found using hte batch gradient descent is close to the one computed using the closed form solution in equation 23.32
