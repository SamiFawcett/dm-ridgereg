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
    for i in range(lower_bound, upper_bound + 1):
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

    return (column_view, point_view, response_header, response_column,
            iv_headers)


def augment(point_view):
    col_view = np.transpose(point_view)
    col_view_ones = np.ones(len(col_view[0]))
    col_view = col_view.tolist()
    col_view.insert(0, col_view_ones.tolist())
    col_view = np.array(col_view)
    augemented_point_view = np.transpose(col_view)

    return augemented_point_view


def ridge_regression_sgd(D, Y, alpha, eta, eps, max_iter):
    t = 0
    aug_D = D
    aug_w = np.ones(len(aug_D[0]))
    converged = False
    prev_w = aug_w
    while not converged or t == max_iter:
        for k in range(1, len(aug_D)):
            r_k = random.randrange(1, len(aug_D))
            p = aug_D[r_k]
            gradient = -((Y[r_k] - np.dot(np.transpose(p), aug_w)) *
                         (p)) + ((alpha / len(aug_D)) * aug_w)

            aug_new_w = np.subtract(aug_w, (eta * gradient))

            prev_w = aug_w
            aug_w = aug_new_w
        t = t + 1
        converged = LA.norm(aug_w - prev_w) <= eps
        if (t == max_iter):
            print('max iterations reached')
            converged = True

    return aug_w


def closed_form_ridge_reg(D, Y, alpha):
    inv = LA.pinv(np.dot(np.transpose(D), D))
    augD_Y = np.dot(np.transpose(D), Y)

    return np.dot(inv, augD_Y)


#data is augmented
def SSE(w, D, Y):
    sse = 0

    for i in range(1, len(D)):
        for j in range(1, len(Y)):
            sse += np.power(Y[j] - np.dot(np.transpose(w), D[i]), 2)

    return sse


def TSS(Y):
    tss = 0
    mu_Y = np.mean(Y)
    for i in range(1, len(Y)):
        tss += np.power((Y[i] - mu_Y), 2)
    return tss


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

    col_view, point_view, resp_header, resp_col, iv_headers = parse_dataset(
        filename)

    #point_views
    training_data = p_range(point_view, 0, 13734)
    validation_data = p_range(point_view, 13735, 13734 + 2000)
    testing_data = p_range(point_view, 13735 + 2000, 13734 + 2000 + 4000)

    training_response = p_range(resp_col, 0, 13734)
    validation_response = p_range(resp_col, 13735, 13734 + 2000)
    testing_response = p_range(resp_col, 13735 + 2000, 13734 + 2000 + 4000)

    closed_form_w = closed_form_ridge_reg(augment(training_data),
                                          training_response, alpha)

    learned_w = ridge_regression_sgd(augment(training_data), resp_col, alpha,
                                     eta, eps, max_iter)

    #find best alpha and eta
    t_alpha = 1
    optimal_alpha = 907

    aug_T = augment(testing_data)

    regularized_w = ridge_regression_sgd(aug_T, testing_response,
                                         optimal_alpha, eta, eps, max_iter)

    tss = TSS(testing_response)
    sse = SSE(closed_form_w, aug_T, testing_response)
    r_squared = (tss - sse) / tss

    print("closed form:", closed_form_w)
    print("optimal alpha:", optimal_alpha)
    print("eta:", eta)
    print("eps:", eps)
    print("ridge regularization sgd learned:", regularized_w)
    print("r^2:", r_squared)
