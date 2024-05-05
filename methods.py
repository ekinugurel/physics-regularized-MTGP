import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees); in meters
    """
    # Convert latitude and longitude to radians
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Calculate the difference between latitudes and longitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Apply the haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r * 1000 # Output distance in meters

def addDist(data, lat = 'lat', lon = 'lng'):
    # Calculate distance between each point and the next point
    lat1, lon1 = data[lat], data[lon]
    # Shift lat/lon columns by 1 row to get the next point
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dist = haversine_np(lat1, lon1, lat2, lon2).fillna(0)
    data['dist'] = dist
    return data

def addVel(data ,lat = 'lat', lon = 'lng', unix = 'unix'):
    lat1, lon1 = data[lat], data[lon]
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dist = data['dist'].fillna(0)
    time_diff = (data[unix] - data[unix].shift(1)).fillna(0)
    vel = dist / time_diff
    vel.iloc[0] = 0
    vel.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinite values with NaN
    vel.fillna(0, inplace=True)
    data['vel'] = vel
    return data

def addBearing(data, lat='lat', lon='lng', verbose=False):
    """
    Calculates the bearing between two points

    Parameters
    ----------
    lat: 
        column containing latitudes
    lon:
        column containing longitudes
    verbose:
        boolean, whether to print out the progress of the function
    """
    if verbose:
        print("Adding bearing column to dataframe...")
    lat1, lon1 = data[lat], data[lon]
    lat2, lon2 = lat1.shift(-1), lon1.shift(-1)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    bearing.iloc[-1] = 0
    data['bearing'] = bearing
    return data

def find_optimal_trip_clusters(data, max_clusters, plot=False):
    # Initialize empty lists to store the inertia values and number of clusters
    inertia_values = []
    num_clusters = []

    # Iterate over different numbers of clusters
    # 1 cluster should not be an option, so we start from 2
    for k in range(2, max_clusters + 1):
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0)  # Choose the number of clusters
        kmeans.fit(data[['latitudeStart', 'longitudeStart', 'latitudeEnd', 'longitudeEnd']])
        data['cluster'] = kmeans.labels_
        
        # Append the inertia value and number of clusters to the lists
        inertia_values.append(kmeans.inertia_)
        num_clusters.append(k)

    if plot:
        # Plot the scree plot
        plt.plot(num_clusters, inertia_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Scree Plot')
        plt.show()

    # Calculate the differences in inertia values
    diff_inertia = np.diff(inertia_values)

    # The elbow is the point where the ratio of consecutive differences is the smallest
    diff_ratio = diff_inertia[:-1] / diff_inertia[1:]

    # Find the index of the elbow point
    elbow_index = np.argmax(diff_ratio) + 1

    # Return the optimal number of clusters
    return elbow_index