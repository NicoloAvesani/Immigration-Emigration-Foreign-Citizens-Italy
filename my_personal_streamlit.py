import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


st.title('NICOLO AVESANI VR490189 SOCIAL RESEARCH FINAL PROJECT 2022-2023')

st.header('DATASET')

italy_emi_data = pd.read_excel('/Users/ave/Desktop/social_research/Italy.xlsx')
italy_emi_data.replace(['..'],0, inplace = True)

# emigrants

italy_emi_data = italy_emi_data[italy_emi_data['Type'] == 'Emigrants']
italy_emi_data = italy_emi_data.drop(italy_emi_data.index[-2:])

italy_emi_data.rename(columns={'OdName':'Country'}, inplace=True)

italy_emi_data = italy_emi_data[italy_emi_data.Country != 'Italy']

for i in italy_emi_data.index:
  if italy_emi_data['Country'][i] == 'China (including Hong Kong Special Administrative Region)':
    italy_emi_data['Country'][i] = 'China'

# drop some useless columns
italy_emi_data = italy_emi_data.drop(['AREA', 'REG', 'DEV'], axis=1)

# add the column total
italy_emi_data['Total'] = italy_emi_data.sum(axis=1)

# drop some columns years
italy_emi_data = italy_emi_data.drop(columns = [1980,       1981,       1982,       1983,       1984,       1985,
             1986,       1987,       1988,       1989,       1990,       1991,
             1992,       1993,       1994,     2001,])

# sort by total
df_sorted_emi = italy_emi_data.sort_values(by='Total', ascending=False)

for i in df_sorted_emi.index:
  if df_sorted_emi['AreaName'][i] == 'Latin America and the Caribbean':
    df_sorted_emi['AreaName'][i] = 'South America'
  elif df_sorted_emi['AreaName'][i] == 'Northern America':
    df_sorted_emi['AreaName'][i] = 'North America'


st.dataframe(df_sorted_emi)


def get_region_input():
    region_options = ['Asia', 'Europe', 'North America', 'South America', 'Africa']
    region = st.sidebar.selectbox('Select a region', region_options)
    return region

region = get_region_input()
st.title('Italian Foreign Emigrates to '+ region +' between 1995 and 2013')


df_area_mask = df_sorted_emi['AreaName'] == region
df_area = df_sorted_emi[df_area_mask]

country_list_world = df_area['Country']
total_list_world = df_area['Total']

data = {
        'Country': country_list_world,
        'Value': total_list_world
    }

df = pd.DataFrame(data)

fig = px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        scope=region.lower(),
        color='Value',
        color_continuous_scale='Viridis',
        range_color=(0, df['Value'].max()),
        labels={'Value': 'Value'},
        title='Foreign Emigrants from Italy between 1995 and 2013 from ' + region
    )

fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False),
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

st.plotly_chart(fig)

# let's see the main country
def get_year_input():
    year = st.sidebar.slider('Select a year', min_value=1995, max_value=2013)
    return year

year = get_year_input()
st.title('Top 10 Destination Countries of Italian Foreign Emigrants in '+ str(year))


fig_1 = plt.figure(figsize=(12, 8))
sb.set(style="white")

df_sorted_year = df_sorted_emi.sort_values(by=year, ascending=False)

sb.barplot(x=df_sorted_year[year].head(10), y=df_sorted_year['Country'].head(10),
               palette="Blues_r", edgecolor=".2")

st.pyplot(fig_1)


top_10_year_input = df_sorted_year.head(10)

country_list = list(top_10_year_input['Country'])
total_list = list(top_10_year_input[year])

# other map of the top 10

data_2 = {
    'Country': country_list,
    'Value': total_list
}

df_2 = pd.DataFrame(data_2)

st.title('Top 10 destionation Countries of Italian Foreign Emigrants in '+ str(year))

fig_2 = px.choropleth(
    df_2,
    locations='Country',
    locationmode='country names',
    color='Value',
    color_continuous_scale='Viridis',
    range_color=(0, df_2['Value'].max()),
    labels={'Value': 'Value'},
    scope='world'
)
fig_2.update_layout(
    geo=dict(showframe=False, showcoastlines=False),
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

st.plotly_chart(fig_2)

country_list_global = list(df_sorted_emi['Country'])
total_list_global = list(df_sorted_emi['Total'])

# plot the global countries per immigration
data_3 = {
    'Country': country_list_global,
    'Value': total_list_global
}

df_3= pd.DataFrame(data_3)

st.title('World destination of Italian Foreign Emigrants')
fig_3 = px.choropleth(
    df_3,
    locations='Country',
    locationmode='country names',
    color='Value',
    color_continuous_scale='Viridis',
    range_color=(0, df_3['Value'].max()),
    labels={'Value': 'Value'}
)


fig_3.update_layout(
    geo=dict(showframe=False, showcoastlines=False),
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

st.plotly_chart(fig_3)

st.title("Italian Foreign Emigrants per Year")

# Specify the video file path
video_path = '/Users/ave/Desktop/social_research/Number of Foreigner Emigrants from Italy per Year.mp4'

# Display the video
st.video(video_path)

st.title("Total Italian Foreign Emigrants 1995-2013")

# Specify the video file path
video_path = '/Users/ave/Desktop/social_research/total_emigrandt_italy_with_sum-2.mp4'

# Display the video
st.video(video_path)

# models
years_int = list(range(1996, 2001)) + list(range(2002, 2014))
tot = pd.DataFrame(italy_emi_data[years_int].sum(axis=0))
tot.index = map(int, tot.index)
tot.reset_index(inplace = True)
tot.columns = ['year', 'total']

#figure 5 
st.title('Total Italian Foreign Emigrates 1995-2013')

fig_5, ax = plt.subplots(figsize=(6, 6))
ax.plot(tot['year'], tot['total'], label='Total Emigration')

# Convert the 'year' and 'total' columns to NumPy arrays
years = tot['year'].values
total = tot['total'].values

# Calculate the coefficients of a polynomial fit
coefficients = np.polyfit(years, total, 1)  # Use 1 for a linear trend line

# Generate the trend line values
trend_line = np.polyval(coefficients, years)

ax.plot(years, trend_line, color='red', label='Trend Line')
ax.set_title('Total Italian Foreign Emigrants from 1995 to 2013')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Immigrants')
ax.legend()

st.pyplot(fig_5)

# regression

x = tot['year']
y = tot['total']
fit = np.polyfit(x, y, deg=1)

fig_6 = plt.figure(figsize=(6, 6))
plt.scatter(tot['year'], tot['total'])
plt.title('Total Emigration 1995-2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

# Assuming you have defined 'x' and 'fit' appropriately
plt.plot(x, fit[0] * x + fit[1], color='red')

st.pyplot(fig_6)


st.title('Polynomial Regression, Predicting the Emigrants')
# polynomial regression
fit = np.polyfit(x, y, deg=1)

# Generate x values for prediction (next years)
x_pred = np.arange(min(x), max(x) + 1)

# Predict the corresponding y values using the fitted line
y_pred = fit[0] * x_pred + fit[1]

fig_7 = plt.figure(figsize=(6, 6))
# Plot the original data points
plt.scatter(x, y, color='blue', label='Original Data')

# Plot the fitted line
plt.plot(x_pred, y_pred, color='red', label='Fitted Line')

# Plot the predicted values for next years
plt.scatter(x_pred, y_pred, color='green', label='Predicted Data')

plt.title('Prediction of Number of Emigrants')
plt.xlabel('Year')
plt.ylabel('Total Immigrants')
plt.legend()

st.pyplot(fig_7)


def get_degree_input():
    degree_options = [1,2,3,4,5,6,7,8,9,10]
    degree = st.sidebar.selectbox('Select a degree', degree_options)
    return degree

degree = get_degree_input()

# fig_8--> linear regression prediction and r2
# Perform polynomial regression
degree = degree  # Adjust the degree of the polynomial as needed
coefficients = np.polyfit(x, y, deg=degree)
poly = np.poly1d(coefficients)

# Generate x values for prediction (including future years)
x_pred = np.arange(min(x), max(x) + 5)  # Extend by 5 years

# Predict the corresponding y values using the polynomial regression
y_pred = poly(x_pred)

fig_8 = plt.figure(figsize=(6, 6))
# Plot the original data points
plt.scatter(x, y, color='blue', label='Original Data')

# Plot the fitted line
plt.plot(x_pred, y_pred, color='red', label='Fitted Line')

# Plot the predicted values for future years
plt.scatter(x_pred[-11:], y_pred[-11:], color='green', label='Predicted Data')

plt.title('Prediction of Number of Emigrants')
plt.xlabel('Year')
plt.ylabel('Total Emigrants')
plt.legend()


# Calculate R-squared score for the polynomial regression
y_pred_train = poly(x)

st.pyplot(fig_8)

#pie chart 
continents = italy_emi_data.groupby('AreaName', axis=0).sum()
print(type(italy_emi_data.groupby('AreaName', axis=0)))
continents_t = continents.T.drop(columns=['World'])
continents = continents_t.T

# Create a new DataFrame for the pie chart
pie_df = continents[['Total']].copy()
pie_df.reset_index(inplace=True)

# Set up the colors and explode list
colors_list = ['green', 'red', 'yellow', 'blue', 'orange', 'black']
explode_list = [0.1, 0.1, 0, 0.1, 0.1, 0]

# Create the interactive pie chart using Plotly
fig_9 = px.pie(pie_df, values='Total', names='AreaName', color_discrete_sequence=colors_list,
             title='Emigration to Italy by Continent [1995 - 2013]',
             hover_data={'Total': ':.1f%'})

# Add percentage labels
fig_9.update_traces(textposition='inside', textinfo='percent+label')

# Update the layout
fig_9.update_traces(hoverinfo='label', marker=dict(line=dict(color='#000000', width=2)))


st.plotly_chart(fig_9)