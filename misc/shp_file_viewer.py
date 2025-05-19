import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# Variable to control sampling rate
time_step = 5

# Load the shapefile
shapefile_path = r"C:\Users\johan\programming\Simulation\misc\radiation_data\git_ignored\ODL\ODL.shp"
gdf = gpd.read_file(shapefile_path)

# Debug statements to check data integrity
if gdf['Time'].isnull().any():
    print("Error: 'Time' column contains null values.")
if gdf['Value'].isnull().any():
    print("Error: 'Value' column contains null values.")
if gdf['geometry'].isnull().any():
    print("Error: 'geometry' column contains null values.")

# Filter data to use every x-th time entry
unique_times = gdf['Time'].unique()
filtered_times = unique_times[::time_step]
filtered_gdf = gdf[gdf['Time'].isin(filtered_times)]

# Prepare data for Plotly
filtered_gdf['Time'] = filtered_gdf['Time'].astype(str)  # Convert Time to string for Plotly slider
fig = px.choropleth(filtered_gdf,
                    geojson=filtered_gdf.geometry,
                    locations=filtered_gdf.index,
                    color='Value',
                    animation_frame='Time',
                    projection="mercator")

# Update layout for better visualization
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title="Polygon Values Over Time",
                  sliders=[{
                      'steps': [{'args': [[t], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                                 'label': t, 'method': 'animate'} for t in filtered_gdf['Time'].unique()],
                      'transition': {'duration': 300},
                      'x': 0.1,
                      'xanchor': 'left',
                      'y': 0,
                      'yanchor': 'top'
                  }])

# Show the figure
fig.show()