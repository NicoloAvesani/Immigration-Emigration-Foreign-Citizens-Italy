import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import statsmodels as sm
import sklearn
from sklearn.metrics import mean_squared_error, r2_score

st.title('Foreign Immigrants to Italy 1995-2013')
st.header('NICOLO AVESANI VR490189 SOCIAL RESEARCH FINAL PROJECT 2022-2023')


st.title('Emigration Dataset')

st.write('In this section, we can see the composition of the Cleaned Dataset for Foreign Immigrants to Italy.')
st.write(" ## The columns are:\n\n'**Type**': 'Immigrants', cause we are considering the Immigrants,\n\n'**Coverage**': tell me if the immigrants are from Italy or from Other countries,\n\n'**Country**': tells me the Birth-Country of the Immigrants, \n\n'**AreaName**': tells me World-Area of the Immigrants, \n\n'**RegName**': tells me the Region Name of origin of the Immigrants, \n\n'**DevName**': tells me if the 'Country' is a developed or developing one, \n\n**Years (1995-2013)**: Immigration from different Countries in the particular year, \n\n'**Total**': tells me the total number of Immigrants between 1995 and 2013 of the particular Country ")

italy_imm_data = pd.read_excel('/Users/ave/Desktop/social_research/Italy.xlsx')
italy_imm_data.replace(['..'],0, inplace = True)

# Immigrants

italy_imm_data = italy_imm_data[italy_imm_data['Type'] == 'Immigrants']
italy_imm_data = italy_imm_data.drop(italy_imm_data.index[-2:])

italy_imm_data.rename(columns={'OdName':'Country'}, inplace=True)

italy_imm_data = italy_imm_data[italy_imm_data.Country != 'Italy']

for i in italy_imm_data.index:
  if italy_imm_data['Country'][i] == 'China (including Hong Kong Special Administrative Region)':
    italy_imm_data['Country'][i] = 'China'

# drop some useless columns
italy_imm_data = italy_imm_data.drop(['AREA', 'REG', 'DEV'], axis=1)

# add the column total
italy_imm_data['Total'] = italy_imm_data.sum(axis=1)

# drop some columns years
italy_imm_data = italy_imm_data.drop(columns = [1980,       1981,       1982,       1983,       1984,       1985,
             1986,       1987,       1988,       1989,       1990,       1991,
             1992,       1993,       1994,     2001,])

# sort by total
df_sorted_emi = italy_imm_data.sort_values(by='Total', ascending=False)

for i in df_sorted_emi.index:
  if df_sorted_emi['AreaName'][i] == 'Latin America and the Caribbean':
    df_sorted_emi['AreaName'][i] = 'South America'
  elif df_sorted_emi['AreaName'][i] == 'Northern America':
    df_sorted_emi['AreaName'][i] = 'North America'


st.dataframe(df_sorted_emi)

try_df = df_sorted_emi.copy()

#global
country_list_global = list(try_df['Country'])
total_list_global = list(try_df['Total'])

# plot the global countries per immigration
data_3 = {
    'Country': country_list_global,
    'Value': total_list_global
}

df_3= pd.DataFrame(data_3)

# world
st.title('World Birth-Countries of Foreign Immigrants to Italy')
fig_3 = px.choropleth(
    df_3,
    locations='Country',
    locationmode='country names',
    color='Value',
    color_continuous_scale='Viridis',
    range_color=(0, df_3['Value'].max()),
    labels={'Value': 'Value'}
)

fig_3.update_geos(showcountries = True)
fig_3.update_layout(
    geo=dict(showframe=True, showcoastlines=False),
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

st.plotly_chart(fig_3)



def get_region_input():
    region_options = ['Asia', 'Europe', 'North America', 'South America', 'Africa']
    region = st.sidebar.selectbox('Select a region for the Map', region_options)
    return region

region = get_region_input()
st.title('Foreign Immigrants to Italy from '+ region +' between 1995 and 2013')


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
        title='Foreign Immigrants to Italy between 1995 and 2013 from ' + region
    )

fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False),
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

st.plotly_chart(fig)
st.write('You can **change** the Region selecting the desired one in the Sidebox')

st.title('Top 10 Birth-Countries of Foreign Immigrants to Italy')
st.write('Select the desired year to analyze')
# let's see the main country
def get_year_input():
    year = st.slider('Select a year', min_value=1995, max_value=2013)
    return year

year = get_year_input()



import plotly.graph_objects as go

df_sorted_year = df_sorted_emi.sort_values(by=year, ascending=False).head(10)

fig_1 = px.bar(df_sorted_year, x='Country', y=year,
               hover_data=['Country', year], color=year,
               labels={year: year, 'Country': 'Countries'},
               template='plotly_white')

fig_1.update_layout(
    title='Top 10 Birth_countries of Foreign Immigrants to Italy',
    xaxis_title='Countries',
    yaxis_title='Immigrants',
)

st.plotly_chart(fig_1)



top_10_year_input = df_sorted_year.head(10)

country_list = list(top_10_year_input['Country'])
total_list = list(top_10_year_input[year])

# other map of the top 10

data_2 = {
    'Country': country_list,
    'Value': total_list
}

df_2 = pd.DataFrame(data_2)



fig_2 = px.choropleth(
    df_2,
    locations='Country',
    locationmode='country names',
    scope="world",
    color='Value',
    color_continuous_scale='Viridis',
    range_color=(0, df_2['Value'].max()),
    labels={'Value': 'Value'},
    title= 'Top 10 Birth-Countries of Foreign Immigrants to Italy in '+ str(year)
)

fig_2.update_geos(showcountries = True)
fig_2.update_layout(
    geo=dict(showframe=False, showcoastlines=False),
    margin={"r": 0, "t": 30, "l": 0, "b": 0}
)

st.plotly_chart(fig_2)

year_chosen_top_5 = top_10_year_input.sort_values(year).tail(5)

pie_df_2 = year_chosen_top_5[['Country', year]].copy()
pie_df_2.reset_index(inplace=True, drop=True)

# Set up the colors and explode list
colors_list = ['green', 'red', 'yellow', 'blue', 'orange']
explode_list = [0.1, 0.1, 0, 0.1, 0.1]

# Create the interactive pie chart using Plotly
fig_12 = px.pie(pie_df_2, values=year, names='Country', color_discrete_sequence=colors_list,
             title='Top 5 Birth-Countries of Foreign Immigrants to Italy in '+str(year),
             hover_data={'Country': ':.1f%'})
# Add percentage labels
fig_12.update_traces(textposition='inside', textinfo='percent+label')

# Update the layout
fig_12.update_traces(hoverinfo='label', marker=dict(line=dict(color='#000000', width=2)))

st.plotly_chart(fig_12)

#pie chart 

st.title('Foreign Immigrants to Italy by Continent 1995-2013')
continents = italy_imm_data.groupby('AreaName', axis=0).sum()
print(type(italy_imm_data.groupby('AreaName', axis=0)))
continents_t = continents.T
continents = continents_t.T

# Create a new DataFrame for the pie chart
pie_df = continents[['Total']].copy()
pie_df.reset_index(inplace=True)

# Set up the colors and explode list
colors_list = ['green', 'red', 'yellow', 'blue', 'orange', 'black']
explode_list = [0.1, 0.1, 0, 0.1, 0.1, 0]

# Create the interactive pie chart using Plotly
fig_9 = px.pie(pie_df, values='Total', names='AreaName', color_discrete_sequence=colors_list,
             title='Total Immigration in Italy by Continent [1995 - 2013]',
             hover_data={'Total': ':.1f%'})

# Add percentage labels
fig_9.update_traces(textposition='inside', textinfo='percent+label')

# Update the layout
fig_9.update_traces(hoverinfo='label', marker=dict(line=dict(color='#000000', width=2)))


st.plotly_chart(fig_9)


def get_region_2_input():
    region_options_2 = ['Asia', 'Europe', 'Latin America and the Caribbean', 'Africa']
    region_2 = st.sidebar.selectbox('Select a region for the Pie', region_options_2)
    return region_2

region_2 = get_region_2_input()

st.title('Foreign Immigrants to Italy from '+region_2+' 1995-2013')

region_chosen = italy_imm_data['AreaName'] == str(region_2)
region_chosen_df = italy_imm_data[region_chosen]

region_chosen_top_5 = region_chosen_df.sort_values('Total').tail(5)

pie_df_2 = region_chosen_top_5[['Country', 'Total']].copy()
pie_df_2.reset_index(inplace=True, drop=True)

# Set up the colors and explode list
colors_list = ['green', 'red', 'yellow', 'blue', 'orange']
explode_list = [0.1, 0.1, 0, 0.1, 0.1]

# Create the interactive pie chart using Plotly
fig_10 = px.pie(pie_df_2, values='Total', names='Country', color_discrete_sequence=colors_list,
             title='Foreign Immigrants to Italy from '+ region_2+' Countries [1995 - 2013]',
             hover_data={'Total': ':.1f%'})

# Add percentage labels
fig_10.update_traces(textposition='inside', textinfo='percent+label')

# Update the layout
fig_10.update_traces(hoverinfo='label', marker=dict(line=dict(color='#000000', width=2)))

st.plotly_chart(fig_10)


st.title("Foreign Immigrants to Italy per Year")

# Specify the video file path
video_path = '/Users/ave/Desktop/social_research/Number of Immigrants arriving in Italy per Year.mp4'

# Display the video
st.video(video_path)

st.title("Total Foreign Immigrants to Italy 1995-2013")

# Specify the video file path
video_path = '/Users/ave/Desktop/social_research/Total arrived in Italy sum.mp4'

# Display the video
st.video(video_path)

# the particular part
st.title('Country Analysis')
st.write('Select the desired Country in the Selectbox')

def country_input():
    region_options = list(df_sorted_emi['Country'])
    country_input = st.selectbox('Select a Country for the Analysis', region_options)
    return country_input

country_input = country_input()
country_mask = df_sorted_emi['Country'] == country_input
country_df = df_sorted_emi[country_mask]

country_df = country_df.drop(columns =['Type','Coverage','Country','AreaName','RegName','DevName','Total'])

st.dataframe(country_df)

df_plot_country = country_df.T
df_plot_country = df_plot_country.rename_axis('Year', axis='index')
df_plot_country = df_plot_country.rename_axis(country_input, axis='columns')

y = df_plot_country.columns
fig_country = px.line(df_plot_country, x=df_plot_country.index, y=y, title='Time Series Growth of Foreign Immigrants to Italy from '+country_input)
st.plotly_chart(fig_country)

#bar plotly chart
fig_country_bar = px.bar(df_plot_country, x=df_plot_country.index, y=y, title='Bar Chart of Foreign Immigrants to Italy from ' + country_input)

# Display the plot using Streamlit
st.plotly_chart(fig_country_bar)

#bar chart growth
growth_input_country = []

year_columns_input_country = list(country_df.columns)

for i in range(len(year_columns_input_country)):
    if i == 0:
        continue
    else:

        growth_input_country.append(((country_df[year_columns_input_country[i]] - country_df[year_columns_input_country[i-1]])/country_df[year_columns_input_country[i-1]])*100)
values = [item.values[0] for item in growth_input_country]
years_to_input = [str(year) for year in range(1996, 2014) if year != 2001]

fig_13 = plt.figure(figsize=(13,8))
plt.title('Percentual growth over years of Foreign Immigrants to Italy from ' + country_input)
data = values
colors = ['red' if x < 0 else 'blue' for x in data]
indices = np.arange(len(data))

plt.bar(indices, data, color=colors)

# Add variation percentages below each bar for negative values
for i, value in enumerate(data):
    if np.isnan(value):
        continue
    if value < 0:
        plt.text(i, value, f'{value:.2f}%', ha='center', va='top')
    else:
        plt.text(i, value, f'{value:.2f}%', ha='center', va='bottom')
plt.xticks(indices, years_to_input)
plt.xlabel('Year')
plt.ylabel('Percentual growth')
st.pyplot(fig_13)

# models
years_int = list(range(1996, 2001)) + list(range(2002, 2014))
tot = pd.DataFrame(italy_imm_data[years_int].sum(axis=0))
tot.index = map(int, tot.index)
tot.reset_index(inplace = True)
tot.columns = ['year', 'total']

#figure 5 
st.title('Total Foreign Immigrants to Italy 1995-2013')

fig_5, ax = plt.subplots(figsize=(6, 6))
ax.plot(tot['year'], tot['total'], label='Total Immigration')

# Convert the 'year' and 'total' columns to NumPy arrays
years = tot['year'].values
total = tot['total'].values

# Calculate the coefficients of a polynomial fit
coefficients = np.polyfit(years, total, 1)  # Use 1 for a linear trend line

# Generate the trend line values
trend_line = np.polyval(coefficients, years)

ax.plot(years, trend_line, color='red', label='Trend Line')
ax.set_title('Total Foreign Immigrants to Italy from 1995 to 2013')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Immigrants')
ax.legend()

st.pyplot(fig_5)


# linear regression
x = tot['year']
y = tot['total']
fit = np.polyfit(x, y, deg=1)

fig_6 = plt.figure(figsize=(10, 6))
plt.scatter(tot['year'], tot['total'])
plt.title('Total Emigration 1995-2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')


# Assuming you have defined 'x' and 'fit' appropriately
plt.plot(x, fit[0] * x + fit[1], color='red')

st.pyplot(fig_6)


st.title('Linear Regression for Predicting the Immigrants')
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

plt.title('Prediction of Number of Immigrants')
plt.xlabel('Year')
plt.ylabel('Total Emigrants')
plt.legend()

st.pyplot(fig_7)
r2 = r2_score(y, fit[0] * x + fit[1])
st.write("R-squared Score:", r2)


st.title('Linear and Polynomial Regression for Predicting the Immigrants')
#polynomial
def get_degree_input():
    degree_options = [1,2,3,4,5,6,7,8,9,10]
    degree = st.selectbox('Select a degree', degree_options)
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

fig_8 = plt.figure(figsize=(6,6))
# Plot the original data points
plt.scatter(x, y, color='blue', label='Original Data')

# Plot the fitted line
plt.plot(x_pred, y_pred, color='red', label='Fitted Line')

# Plot the predicted values for future years
plt.scatter(x_pred[-11:], y_pred[-11:], color='green', label='Predicted Data')

plt.title('Prediction of Number of Immigrants')
plt.xlabel('Year')
plt.ylabel('Total Emigrants')
plt.legend()
plt.show()

st.pyplot(fig_8)

#r2
y_pred_train = poly(x)
r2 = r2_score(y, y_pred_train)
st.write("R-squared Score:", r2)


st.title('Using ARIMA to Predict the Total Immigrants in next 3 years')
# last model, use the stats model ARIMA
import statsmodels.api as sm

df_prediction = tot.copy()
# Prepare the data for time series forecasting
df_prediction['year'] = pd.to_datetime(df_prediction['year'], format='%Y')
df_prediction.set_index('year', inplace=True)

# Create and fit the ARIMA model
model = sm.tsa.ARIMA(df_prediction['total'], order=(1, 1, 1))
model_fit = model.fit()

# Predict immigration for the next three years
next_years = pd.date_range(start='2014', periods=3, freq='A')
predicted_immigration = model_fit.predict(start=df_prediction.shape[0], end=df_prediction.shape[0]+2)

# Plotting the historical data
fig_11 = plt.figure(figsize=(6,6))
years = tot['year'].values
immigration = tot['total'].values

plt.plot(years, immigration, marker='o', linestyle='-', label='Historical Data')

# Plotting the predicted values
next_years = np.arange(2014, 2017)
predicted_immigration = predicted_immigration

plt.plot(next_years, predicted_immigration, marker='o', linestyle='-', label='Predicted Data')

# Adding labels and title to the plot
plt.xlabel('Year')
plt.ylabel('Emigration')
plt.title('Historical and Predicted Emigration')
plt.legend()

# Display the plot
st.pyplot(fig_11)