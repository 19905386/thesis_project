import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import array
import math
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm as tqdm
import cupy as cp
from numba import jit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from collections import defaultdict
# Insert the path of modules folder 
# sys.path.insert(0, 'C:/Users/Jason/thesis_project')
sys.path.append("../")

from support import *
from features.feature_ts import genX
from experiment.algorithms.cluster_prep import *

def range_min3(df, indices_A):
    dy = [df[i+1]-df[i] for i in range(len(df)-1)]

    if indices_A == 11 or indices_A == 0 or indices_A == 12:
        return False  
    else:
        if dy[indices_A - 1] > 0 and dy[indices_A] < 0:
            return True    
        else:
            return False

# Find Peaks Function
def find_peaksMA(X):
    
    df = X.copy()

    arrayY = X.to_numpy().copy()

    listYM  = list(map(lambda x : x[0:12],arrayY))
    listYA  = list(map(lambda x : x[12:],arrayY))

    arrayYM = cp.array(listYM)
    arrayYA = cp.array(listYA)

    arrayYM1 = cp.array(listYM)
    arrayYA1 = cp.array(listYA)

    indices_M = cp.where(arrayYM[0] == cp.amax(arrayYM[0]))
    indices_A = cp.where(arrayYA[0] == cp.amax(arrayYA[0]))

    while range_min3(arrayYM1[0], indices_M[0][0]) == False:
        arrayYM[0][cp.where(arrayYM[0] == cp.amax(arrayYM[0]))] = 0
        indices_M = cp.where(arrayYM[0] == cp.amax(arrayYM[0]))
        # print("arrayYM = {}".format(arrayYM))
        all_zeros = not cp.any(arrayYM[0])
        if all_zeros == True:
            return df, False
        # print("all_zeros = {}".format(all_zeros))
  
    while range_min3(arrayYA1[0], indices_A[0][0]) == False:
        # Find the second largest peak
        arrayYA[0][cp.where(arrayYA[0] == cp.amax(arrayYA[0]))] = 0
        indices_A = cp.where(arrayYA[0] == cp.amax(arrayYA[0]))
        all_zeros = not cp.any(arrayYA[0])
        if all_zeros == True:
            return df, False

    df['0_x'] = indices_M[0] 
    df['0_y'] = indices_A[0] + 12 

    return df, True

# Load data
X = genX([1994,2014], drop_0 = True)

Xbin = xBins(X, 'amd')

# Select a household
def selectRandomUser(Household):
    
    house_id = Household
    
    # Create a dataframe with all of that households data
    A = X.loc[house_id,:]
    H = A.copy()

    return H

# Get households average profile
def describe_household(id):
       # Household ids
       # id = [12020322]#[483]#[2230]#,1002790,483,3063,5351,12020322,5283,5293,168,7118,156,5239,29,1418,1005536]
       id = [id]

       for i in id:
              H = selectRandomUser(i)

              # H1 = find_peaksMA(H)
              H1_d = H.describe()

              H1_d = H1_d.drop(['count','std','min','max', '25%','50%','75%'])
              
              H1_d1, check = find_peaksMA(H1_d)

    #    H1_d1.drop("index", axis = 1, inplace=True)
              
       return H1_d1

#Define the Gaussian function
def gauss(x, H, A, mu, sigma):
    return H + A * cp.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Define function that approximates straight line
def straight_line(m,x,b):
    return m*x+b

# Function that calculates the sigma values of gaussian fits
def Sigma1(increment = 0):
    # Conditional if statement to check that 
    p1 = 4 + increment

    i1 = 0

    Xpeak1 = df[str(df['0_x'].values[0])].values[0]

    for c in df:
        if int(c) != df['0_x'].values[0]:
            x1 = df[str(df['0_x'].values[0] - int(c))].values[0]
            temp1 = Xpeak1 - x1

            if temp1 >= 0.0:
                Xpeak1 = x1
            else:
                i1 = df['0_x'].values[0]-int(c) # i returns the index of the lower boundary of range
                break
        else:
            i1 = df['0_x'].values[0]-int(c) # i returns the index of the lower boundary of range
            break

    range1 = 2*(cp.sqrt((df['0_x'].values[0] - i1)**2))
    
    sigma1 = range1/p1

    return sigma1, i1


def Sigma2(increment = 0):
    # Conditional if statement to check that 
    p1 = 4 + increment

    i1 = 0

    Xpeak1 = df[str(df['0_x'].values[0])].values[0]# Get the value at the peak
    
    for c in df:
        x1 = df[str(df['0_x'].values[0]+int(c))].values[0]
        temp1 = Xpeak1 - x1
        # print("temp1 = Xpeak1 - x1 = {} - {} = {}".format(Xpeak1,x1,temp1))
        if temp1 >= 0:
            Xpeak1 = x1
        else:
            i1 = df['0_x'].values[0]+int(c) # i returns the index of the lower boundary of range
            break

    range1 = 2*(cp.sqrt((df['0_x'].values[0] - i1)**2))

    sigma1 = range1/p1

    return sigma1, i1


def Sigma3(increment = 0):  
    # Conditional if statement to check that 
    p2 = 4 + increment

    # i1 = 0
    i2 = 0

    Xpeak2 = df[str(df['0_y'].values[0])].values[0]

    for c in df:
        x2 = df[str(df['0_y'].values[0]-int(c))].values[0]
        temp2 = Xpeak2 - x2
        
        if temp2 >= 0.0:
            Xpeak2 = x2
        else:
            i2 = df['0_y'].values[0]-int(c)+1 # i returns the index of the lower boundary of range
            break

    range2 = 2*(cp.sqrt((df['0_y'].values[0] - i2)**2))

    sigma2 = range2/p2

    return sigma2, i2

def Sigma4(increment=0):
    # Conditional if statement to check that 
    p2 = 4 + increment

    i2 = 0

    Xpeak2 = df[str(df['0_y'].values[0])].values[0]# Get the value at the peak
    
    for c in df:
        try:
            x2 = df[str(df['0_y'].values[0]+int(c))].values[0] # Get the one after the peak
            temp2 = Xpeak2 - x2
            # print("temp2 = Xpeak2 - x2 = {} - {} = {}".format(Xpeak2,x2,temp2))
            
            if temp2 >= 0:
                Xpeak2 = x2
            else:
                i2 = df['0_y'].values[0]+int(c) # i returns the index of the lower boundary of range
                break
            
        except KeyError:
            i2 = df['0_y'].values[0]+int(c)
            break

    range2 = 2*(cp.sqrt((df['0_y'].values[0] - i2)**2))
    
    sigma2 = range2/p2

    return sigma2, i2


# Create dataframe to store RMSE
list_ids = []
for id in Xbin['50-150']:
    list_ids.append(id)

ids_df = pd.DataFrame(index = list_ids, columns = ['RMSE'])
counter = 0
for id in tqdm(Xbin['50-150']):
    counter = counter + 1
    houseID = id
    df, check = describe_household(id = houseID)

    # Check whether the household has any morning/afternoon peaks
    if check == False:
        continue

    # Determine the y-offset i.e. minimum value in day
    H_offset = df.iloc[0,:24].min()

    # Determine the amplitude of gaussian
    A1 = df[str(df['0_x'].values[0])].values[0] - H_offset # Adjust for offset
    A2 = df[str(df['0_y'].values[0])].values[0] - H_offset

    # Determine the mean of the gaussian
    mu1 = df['0_x'].values[0]
    mu2 = df['0_y'].values[0]
    # **********************************************************************************************************************************************
    # Fit the first gaussian
    sigma1, i1 = Sigma1()

    # Calculate error
    actual = df.iloc[0,i1:df['0_x'].values[0]].to_list()
    predicted = gauss(cp.arange(i1,df['0_x'].values[0],1),H_offset,A1,mu1,sigma1)
    MSE = mean_squared_error(actual, predicted)
    RMSE = cp.sqrt(MSE)

    tolerance1 = RMSE - RMSE*0.95
    i = 0.1 # Constant value by which to increase/decrease the range in the initial sigma function
    t = 0
    while RMSE > tolerance1:
        previous_error1 = RMSE

        sigma1, i1= Sigma1(increment = i)
        predicted = gauss(cp.arange(i1,df['0_x'].values[0],1),H_offset,A1,mu1,sigma1) # Get the value of the gauss one point above the min peak
        MSE = mean_squared_error(actual, predicted)
        RMSE = cp.sqrt(MSE)   
        # conditional for gauss 1
        if RMSE > previous_error1:
            i = i + 0.1
            sigma1, i1 = Sigma1(increment = i)
            predicted = gauss(cp.arange(i1,df['0_x'].values[0],1),H_offset,A1,mu1,sigma1) # Get the value of the gauss one point above the min peak
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE) 
        else:
            i = i - 0.1
            sigma1, i1 = Sigma1(increment = i)
            predicted = gauss(cp.arange(i1,df['0_x'].values[0],1),H_offset,A1,mu1,sigma1) # Get the value of the gauss one point above the min peak
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE) 

        t = t + 1
        if t >= 10000:
            t = 0
            i = 0
            tolerance1 = tolerance1 + 0.1

    RMSE1 = RMSE
    # print("RMSE1 = {}".format(RMSE1))
    sigma1 = cp.sqrt(sigma1**2)
    # **********************************************************************************************************************************************
    # Fit the secnd gaussian
    sigma2, i2= Sigma2()

    # Calculate error
    actual = df.iloc[0,df['0_x'].values[0]:i2].to_list()
    predicted = gauss(cp.arange(df['0_x'].values[0],i2,1),H_offset,A1,mu1,sigma2)
    MSE = mean_squared_error(actual, predicted)
    RMSE = cp.sqrt(MSE)

    tolerance2 = RMSE - RMSE*0.95
    # print(tolerance2)
    i = 0.1 # Constant value by which to increase/decrease the range in the initial sigma function
    t = 0
    while RMSE > tolerance2:

        # print("RMSE = {}".format(RMSE))
        # print("tolerance2 = {}".format(tolerance2))
        # print("actual = {}".format(actual))
        # print("predicted = {}".format(predicted))
        previous_error2 = RMSE
        sigma2, i2 = Sigma2(increment = i)
        predicted = gauss(cp.arange(df['0_x'].values[0],i2,1),H_offset,A1,mu1,sigma2)
        MSE = mean_squared_error(actual, predicted)
        RMSE = cp.sqrt(MSE)
        
        # Conditional for gauss2
        if RMSE > previous_error2:
            i = i + 0.1
            sigma2, i2 = Sigma2(increment = i)
            predicted = gauss(cp.arange(df['0_x'].values[0],i2,1),H_offset,A1,mu1,sigma2)
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE)
        else:
            i = i -0.1        
            sigma2, i2 = Sigma2(increment = i)
            predicted = gauss(cp.arange(df['0_x'].values[0],i2,1),H_offset,A1,mu1,sigma2)
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE)

        t = t + 1
        if t >= 1000:
            t = 0
            i = 0
            tolerance2 = tolerance2 + 0.1


    RMSE2 = RMSE
    # print("RMSE2 = {}".format(RMSE2))
    sigma2 = cp.sqrt(sigma2**2)
    # **********************************************************************************************************************************************
    # Fit the third gaussian
    # Code that finds the value one point above the range index
    sigma3, i3 = Sigma3()

    # Calculate error
    actual = df.iloc[0,i3:df['0_y'].values[0]].to_list()
    predicted = gauss(cp.arange(i3,df['0_y'].values[0],1),H_offset,A2,mu2,sigma3)
    MSE = mean_squared_error(actual, predicted)
    RMSE = cp.sqrt(MSE)

    tolerance1 = RMSE - RMSE*0.95
    i = 0.1 # Constant value by which to increase/decrease the range in the initial sigma function
    t = 0
    while RMSE > tolerance1:
        previous_error3 = RMSE
        # print(sigma3)
        # print("RMSE = {}".format(RMSE))

        sigma3, i3 = Sigma3(increment = i)
        predicted = gauss(cp.arange(i3,df['0_y'].values[0],1), H_offset, A2, mu2, sigma3)
        # print('actual = {}'.format(actual))
        # print('predicted = {}'.format(predicted))
        MSE = mean_squared_error(actual, predicted)
        RMSE = cp.sqrt(MSE)

        # conditional for gauss 1
        if RMSE > previous_error3:
            i = i + 0.1
            sigma3, i3 = Sigma3(increment = i)
            predicted = gauss(cp.arange(i3,df['0_y'].values[0],1),H_offset,A2,mu2,sigma3)
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE)
        else:
            i = i - 0.1
            sigma3, i3 = Sigma3(increment = i)
            predicted = gauss(cp.arange(i3,df['0_y'].values[0],1),H_offset,A2,mu2,sigma3)
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE)

        t = t + 1
        if t >= 10000:
            t = 0
            i = 0
            tolerance1 = tolerance1 + 0.1
    

    RMSE3 = RMSE
    # print("RMSE3 = {}".format(RMSE))
    sigma3 = cp.sqrt(sigma3**2)
    # **********************************************************************************************************************************************
    # Fit the 4th gaussian
    sigma4, i4= Sigma4()

    actual = df.iloc[0,df['0_y'].values[0]:i4].to_list()
    predicted = gauss(cp.arange(df['0_y'].values[0],i4,1),H_offset,A2,mu2,sigma4)
    MSE = mean_squared_error(actual, predicted)
    RMSE = cp.sqrt(MSE)

    tolerance2 = RMSE - RMSE*0.95
    # print(tolerance2)
    i = 0.1 # Constant value by which to increase/decrease the range in the initial sigma function
    t = 0
    while RMSE > tolerance2:
        previous_error4 = RMSE
        # print('sigma4 = {}'.format(sigma4))
        # print('RMSE = {}'.format(RMSE))
        sigma4, i4 = Sigma4(increment = i)
        predicted = gauss(cp.arange(df['0_y'].values[0],i4,1),H_offset,A2,mu2,sigma4)
        MSE = mean_squared_error(actual, predicted)
        RMSE = cp.sqrt(MSE)
        
        # Conditional for gauss2
        if RMSE > previous_error4:
            i = i + 0.1
            sigma4, i4 = Sigma4(increment = i)
            predicted = gauss(cp.arange(df['0_y'].values[0],i4,1),H_offset,A2,mu2,sigma4)
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE)

        else:
            i = i - 0.1        
            sigma4, i4 = Sigma4(increment = i)
            predicted = gauss(cp.arange(df['0_y'].values[0],i4,1),H_offset,A2,mu2,sigma4)
            MSE = mean_squared_error(actual, predicted)
            RMSE = cp.sqrt(MSE)

        t = t + 1
        if t >= 10000:
            t = 0
            i = 0
            tolerance2 = tolerance2 + 0.1
    # print(tolerance2)
    RMSE4 = RMSE
    sigma4 = cp.sqrt(sigma4**2)
    # **********************************************************************************************************************************************
    # Determine the complete synthetic profile
    x_1 = cp.arange(0, mu1, 1)
    x_2 = cp.arange(mu1, i2, 1)
    x_3 = cp.arange(i3, mu2,1)
    x_4 = cp.arange(mu2, 24,1)

    x_vals = cp.hstack((x_1,x_2,x_3,x_4))
    x_vals_1 = cp.unique(x_vals)

    y1 = gauss(x_1, H_offset,A1,mu1,sigma1)
    y2 = gauss(x_2, H_offset,A1,mu1,sigma2)
    y3 = gauss(x_3, H_offset,A2,mu2,sigma3)
    y4 = gauss(x_4, H_offset,A2,mu2,sigma4)

    if len(x_vals_1) == 24:
        # print("Entire array is captured")
        if x_2[-1] == x_3[0]:
            # print('repeating value')
            x_3 = x_3[1:] # Remove the first value of the x-array
            y3 = gauss(x_3, H_offset,A2,mu2,sigma3) # Re-calculate the y-values for gauss 3
            synth = cp.hstack((y1,y2,y3,y4)) # Generate the synthetic values
        else:
            synth = cp.hstack((y1,y2,y3,y4)) # If there is no repeating values in x_2 and x_3 then generate synthetic profiles without adjusting x_3's values

    else:
        # print("Missing hours need to be saved")
        x_5 = cp.arange(i2,i3,1)
        x_vals_1 = cp.hstack((x_1,x_2,x_5,x_3,x_4))
        x_vals_1 = cp.unique(x_vals_1)

        # Synthesize the mid range behaviour
        m = (gauss(x_3[0], H_offset,A2,mu2,sigma3) - gauss(x_2[-1], H_offset,A1,mu1,sigma2))/(x_3[0]-x_2[-1]) # Calculate slope
        b = gauss(x_3[0], H_offset,A2,mu2,sigma3) - m*x_3[0] # Calculate b constant for straight line equation
        y_mid = straight_line(m,x_5,b)

        synth = cp.hstack((y1,y2,y_mid,y3,y4))
    # **********************************************************************************************************************************************
    # Calculate the synthetic profiles RMSE
    actual = df.iloc[0,0:24]
    predicted = synth
    MSE = mean_squared_error(actual, predicted)
    RMSE = cp.sqrt(MSE)
    # **********************************************************************************************************************************************
    # Store the RMSE error
    ids_df.loc[id] = RMSE
    # **********************************************************************************************************************************************
    if counter%50 == 0:
        # Save plot of fit
        fig, ax = plt.subplots(figsize=(12,6))
        plt.style.use('fivethirtyeight')  

        # Plot the households loadprofile
        fig = plt.plot(describe_household(id = houseID).iloc[0,0:24], linewidth = 1.5, label = "Measured Profile")#, linestyle = 'dashed')
        plt.title("Household "+str(i))
        x_all = cp.arange(0,24,1)
        # Plot Gauss Fit
        ax.plot(x_all,synth,linewidth = 1.5, color = 'purple',label = "Synthetic Profile, RMSE = " + str('{0:.3g}'.format(RMSE)))

        plt.legend(loc = "upper left")
        # Set the labels and axis limits
        ax.set_xlim([0,24])
        ax.set_xlabel('Time of day')
        ax.set_ylabel("Amplitude")
        ax.set_title('Synthetic Profile: Household ' + str(houseID))

        # # Save the figure
        plt.savefig('Synthetic Profile' + str(houseID) + '.png', dpi=72, bbox_inches='tight')
    # **********************************************************************************************************************************************
    # **********************************************************************************************************************************************
    # **********************************************************************************************************************************************


ids_df.to_csv('RMSE_1200-2500.csv', header=False, index=False)