#!/usr/bin/env python
# coding: utf-8

# # **Assessing the level of hospital services in various states and pointing out areas for improvement.**
# 
# Business-wise, this study could offer hospital management and decision-makers insights into potential areas for development in order to raise the standard of care given to patients. Additionally, it might assist stakeholders and investors in choosing which hospitals or states to fund or invest in.

# In[15]:


# Imports necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import plotly.express as px


# # **Data Importing and Basic Data cleaning**
# 

# In[17]:


# Read the CSV file 'hospital_general_info.csv' and stores it in a Pandas DataFrame
hospital_general_info = pd.read_csv('F:/University of Galway/01_MSc Business Analytics/Semester 2/MS5114 Advanced Programming for Business Analytics/Case Studies and Assignments/Group Assignment/DataSet/Data/hospital_general_info.csv')

# Perform data cleaning on various columns
hospital_general_info = hospital_general_info.replace('Not Available', 0)
hospital_general_info = hospital_general_info.astype(str)
hospital_general_info = hospital_general_info.replace('nan', np.nan)
hospital_general_info = hospital_general_info.fillna(0)


hospital_general_info[['hospital_overall_rating', 'facility_mortaility_measures_count', 'facility_care_safety_measures_count', 'facility_readmission_measures_count', 'patient_experience_measures_footnote', 'facility_timely_and_effective_care_measures_count']]


# Selecting few columns and cleaning it
columns_to_clean = ['hospital_overall_rating', 'facility_mortaility_measures_count', 'facility_care_safety_measures_count', 'facility_readmission_measures_count', 'patient_experience_measures_footnote', 'facility_timely_and_effective_care_measures_count']

for column in columns_to_clean:
    hospital_general_info[column] = hospital_general_info[column].str.replace('nannannannan', '')
    hospital_general_info[column] = hospital_general_info[column].str.replace('.', '')
    hospital_general_info[column] = pd.to_numeric(hospital_general_info[column])

# Now check the updated DataFrame
hospital_general_info.head()


# # **Analyzing and displaying information on hospitals in general**
# 
# In order to compare and identify the states with the greatest and lowest quality measures, the code groups the hospital data by state, calculates the mean values of several quality measures, and generates visualizations like bar charts and heatmaps.

# In[18]:


# The average values of various quality measures were computed for each state using a grouping of the data by state
grouped_data = hospital_general_info.groupby('state').agg({'hospital_overall_rating': 'mean',
                                                              'facility_mortaility_measures_count': 'mean',
                                                              'facility_care_safety_measures_count': 'mean',
                                                              'facility_readmission_measures_count': 'mean',
                                                              'patient_experience_measures_footnote': 'mean',
                                                              'facility_timely_and_effective_care_measures_count': 'mean'})

grouped_data['total_quality'] = grouped_data.sum(axis=1)

lowest_quality_states = grouped_data.sort_values(by='total_quality', ascending=True)
lowest_quality_states = lowest_quality_states.reset_index()

national_averages = hospital_general_info[['hospital_overall_rating',
                                             'facility_mortaility_measures_count',
                                             'facility_care_safety_measures_count',
                                             'facility_readmission_measures_count',
                                             'patient_experience_measures_footnote',
                                             'facility_timely_and_effective_care_measures_count']].mean()
print(national_averages)
print()
print()


state_averages = hospital_general_info.groupby('state').agg({'hospital_overall_rating': 'mean',
                                                              'facility_mortaility_measures_count': 'mean',
                                                              'facility_care_safety_measures_count': 'mean',
                                                              'facility_readmission_measures_count': 'mean',
                                                              'patient_experience_measures_footnote': 'mean',
                                                              'facility_timely_and_effective_care_measures_count': 'mean'})

# Create bar chart
fig, ax = plt.subplots(figsize=(12,6))
state_averages.plot(kind='bar', ax=ax)
ax.axhline(y=national_averages['patient_experience_measures_footnote'], color='red', linestyle='--')
ax.set_xlabel('State')
ax.set_ylabel('Average Value')
ax.legend()
plt.show()


# Select the 'state' and 'total_quality' columns from hospital_general_info
heatmap_data = lowest_quality_states[['state', 'total_quality']]

# Use plotly to create a choropleth map of the US with a heatmap color scale
fig = px.choropleth(locations=heatmap_data['state'], locationmode='USA-states', color=heatmap_data['total_quality'], scope='usa', color_continuous_scale='RdYlGn', range_color=(0, heatmap_data['total_quality'].max()), title='Total Quality by State')
fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgb(255, 255, 255)'),title=dict(x=0.5, xanchor='center'))

# Display the plot
fig.show()


# # **Correlation matrix for a subset of columns**
# 
# In order to find any connections or patterns between the variables in the hospital data that would be helpful in making business choices, developing a correlation matrix would be the appropriate course of action.

# In[6]:


# Creating a correlation matrix for a subset of columns
corr_matrix = hospital_general_info[['hospital_name', 'zip_code', 'hospital_type', 'hospital_ownership', 'hospital_overall_rating', 'facility_mortaility_measures_count', 
                                     'facility_care_safety_measures_count', 'facility_readmission_measures_count', 'facility_timely_and_effective_care_measures_count']].corr()
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix')
plt.show()


# # **K - means clustering on selected variables and principal component analysis (PCA)**
# 
# Grouping hospitals according to their overall rating and zip code using the k-means clustering algorithm, and then uses bar charts to display the mean overall rating and zip code of each cluster. The purpose of this analysis is to learn more about how hospitals are doing around the country and whether there may be a relationship between location and overall rating. This can aid informing decisions regarding resource allocation and quality improvement programs made by healthcare organizations and policymakers. For instance, if hospitals in a given cluster are situated in areas with higher zip codes but have lower overall ratings, it may be necessary to implement focused interventions to raise the standard of treatment in that area. 

# In[8]:


# Read the CSV file 'hospital_general_info.csv' and stores it in a Pandas DataFrame
hospital_general_info = pd.read_csv('F:/University of Galway/01_MSc Business Analytics/Semester 2/MS5114 Advanced Programming for Business Analytics/Case Studies and Assignments/Group Assignment/DataSet/Data/hospital_general_info.csv')

# Select variables of interest
hospital_data_frame = hospital_general_info[['state', 'zip_code', 'hospital_name', 'hospital_overall_rating', 'city', 'hospital_ownership', 'safety_measures_count']]

# Handle missing values
hospital_data_frame.dropna(inplace=True)
hospital_data_frame['safety_measures_count'].fillna(0, inplace=True)
hospital_data_frame.replace('Not Available', 0, inplace=True)

# One-hot encode Ownership variable
hospital_data_frame = pd.get_dummies(hospital_data_frame, columns=['hospital_ownership'])

# Normalize quantitative variables
scaler = StandardScaler()
hospital_data_frame[['zip_code', 'hospital_overall_rating']] = scaler.fit_transform(hospital_data_frame[['zip_code', 'hospital_overall_rating']])

# Apply k-means clustering
hospital_data_frame = hospital_data_frame.drop(['state', 'hospital_name', 'city', 'safety_measures_count'], axis=1)
kmeans = KMeans(n_clusters=4, random_state=0).fit(hospital_data_frame)

# Visualize clusters using PCA and scatter plot
pca = PCA(n_components=2)
hospital_data_frame_pca = pca.fit_transform(hospital_data_frame)
plt.scatter(hospital_data_frame_pca[:, 0], hospital_data_frame_pca[:, 1], c=kmeans.labels_)
plt.xlabel('hospital_overall_rating')
plt.ylabel('zip_code')
plt.show()

# Add 'cluster' column to X
hospital_data_frame['cluster'] = kmeans.labels_


# # **Create a bar chart for selected variables**
# 
# The objective of this analysis, from a commercial perspective, may be to learn more about how hospitals function within various clusters and to pinpoint the elements that support their growth. It is possible to find patterns and trends in the data by grouping hospitals based on their ownership, number of safety measures, zip code, and overall rating. These patterns and trends may be utilized to optimize resource allocation, increase patient care, and improve the reputation of hospitals.

# In[9]:


# Create a list of columns
cols = ['hospital_overall_rating','zip_code']

# Define color range for each cluster
colors = ['lightblue', 'lightgreen', 'lightpink', '#b19cd9', 'purple', 'gray']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(15,5))

# Loop through the columns and plot them on separate axes
for i, col in enumerate(cols):
    # Group by cluster and calculate mean for the column
    cluster_means = hospital_data_frame.groupby('cluster')[col].mean()

    # Plot the bar chart on the corresponding axis
    ax = axes[i]
    ax.bar(cluster_means.index, cluster_means.values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel(col)
    ax.set_title(f'Mean {col} by Cluster')
    
# Adjust the spacing between the plots
plt.tight_layout()

# Show the plots
plt.show()



# # **Create a Box Plot to visualize our clusters**
# 
# The box plots make it easier to see how the various clusters differ in terms of zip codes and overall ratings. For instance, it may be a sign that the hospitals in a cluster are dealing with particular difficulties because of their location or patient base if the hospitals in the cluster have lower overall ratings and have concentrated zip codes. This could guide programs or focused actions meant to raise the efficiency of certain hospitals.

# In[10]:


# Create a new column for the cluster labels
hospital_data_frame['cluster'] = kmeans.labels_

# Create box plots for each variable, grouped by cluster
sns.boxplot(x='cluster', y='zip_code', data=hospital_data_frame)
plt.xlabel('Cluster')
plt.ylabel('Zip Code')
plt.show()

sns.boxplot(x='cluster', y='hospital_overall_rating', data=hospital_data_frame)
plt.xlabel('Cluster')
plt.ylabel('Overall Rating')
plt.show()

