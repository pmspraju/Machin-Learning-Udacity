""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
import numpy as np
def featureScaling(arr):
    arr_min = np.nanmin(np.array(arr).astype(np.float))
    arr_max = np.nanmax(np.array(arr).astype(np.float))
    den = arr_max - arr_min
    scaled_features = map(lambda x: ((x-arr_min)/den), arr)
    return scaled_features

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
