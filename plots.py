import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plotTrip(points_m, trip_id, trip_id_col = 'tid', title='Similar bus trips'):
    # Plot black and white
    trip = points_m[points_m[trip_id_col] == trip_id]
    restOfTrips = points_m[points_m[trip_id_col] != trip_id]
    fig = plt.figure(figsize=(10, 10))
    plt.plot(restOfTrips['lat'], restOfTrips['lng'], label='Other Paths', alpha=0.3, color='black')
    plt.plot(trip['lat'], trip['lng'], label='Path', color='black')
    plt.scatter(restOfTrips['latitudeStart'], restOfTrips['longitudeStart'], label='Other Starts', marker='.', alpha=0.3, color='red')
    plt.scatter(restOfTrips['latitudeEnd'], restOfTrips['longitudeEnd'], label='Other Ends', marker='.', alpha=0.3, color='blue')
    plt.scatter(trip['latitudeStart'], trip['longitudeStart'], label='Start', marker='^', color='red', s=10)
    plt.scatter(trip['latitudeEnd'], trip['longitudeEnd'], label='End', marker='^', color='blue', s=10)
    plt.legend(fontsize=15)
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('Latitude', fontsize=15)
    plt.ylabel('Longitude', fontsize=15)
    plt.show()
    return fig

def plotClusters(data):
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(data['latitudeStart'], data['longitudeStart'], c=data['start_cluster'], cmap='viridis', s=5)
    plt.colorbar()
    plt.title('Trip Starting Points')
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    plt.scatter(data['latitudeEnd'], data['longitudeEnd'], c=data['end_cluster'], cmap='viridis', s=5)
    plt.colorbar()
    plt.title('Trip Ending Points')
    plt.tight_layout()
    plt.show()

def plotTopTwoClusters(top_data):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(top_data['latitudeStart'], top_data['longitudeStart'], c=top_data['cluster'], cmap='viridis', label='Start')
    plt.scatter(top_data['latitudeEnd'], top_data['longitudeEnd'], c=top_data['cluster'], cmap='viridis', marker='x', label='End')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Top Two Clusters')
    plt.legend()
    plt.show()

def trainingLossPlot(ls):
    iters = range(0, len(ls))
    fig4, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(iters, ls, 'g')
    ax.set_title('Training Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    return fig4
    
def plotModelMetricsGeoLife(bike_df,
                            walk_df,
                            bus_df,
                            figsize=(20, 10),
                            fontsize=25,
                            dpi=100,
                            logy=False,
                            xtick_rotation=0):
    plt.figure(figsize=figsize, dpi=dpi)
    # Share y-axis
    #fig, axes = plt.subplots(1, 5, sharey=True, figsize=figsize)
    # Make font size larger
    plt.subplot(1, 3, 1)
    sns.boxplot(data=bike_df, x='Metric', y='value', hue='Model')
    plt.xlabel('', fontsize=fontsize)
    plt.ylabel('', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=xtick_rotation)
    plt.yticks(fontsize=fontsize)
    plt.yscale('log' if logy else 'linear')
    plt.title('Bike Trips', fontsize=fontsize)
    plt.legend(title='Model', fontsize=fontsize, title_fontsize=fontsize)
    #plt.tight_layout()

    plt.subplot(1, 3, 2)
    sns.boxplot(data=walk_df, x='Metric', y='value', hue='Model')
    plt.xlabel('', fontsize=fontsize)
    plt.ylabel('', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=xtick_rotation)
    plt.yticks(fontsize=fontsize)
    plt.yscale('log' if logy else 'linear')
    plt.title('Walk Trips', fontsize=fontsize)
    plt.legend('')
    #plt.tight_layout()

    plt.subplot(1, 3, 3)
    sns.boxplot(data=bus_df, x='Metric', y='value', hue='Model')
    plt.xlabel('', fontsize=fontsize)
    plt.ylabel('', fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=xtick_rotation)
    plt.yticks(fontsize=fontsize)
    plt.yscale('log' if logy else 'linear')
    plt.title('Bus Trips', fontsize=fontsize)
    plt.legend('')
    plt.show()