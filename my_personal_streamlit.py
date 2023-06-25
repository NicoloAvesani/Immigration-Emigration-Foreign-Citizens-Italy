import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sb

st.title('NICOLO AVESANI VR490189 SOCIAL RESEARCH FINAL PROJECT 2022-2023')

st.header('DATASET')

italy_emi_data = pd.read_excel('/Users/ave/Desktop/social_research-1/Italy.xlsx')
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
    region_options = ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Oceania']
    region = st.sidebar.selectbox('Select a region', region_options)
    return region

st.title('Immigrants to Italy between 1995 and 2013')
region = get_region_input()

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
        title='Immigrants to Italy between 1995 and 2013 from ' + region
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


st.title('Top 10 Countries by Emigrants from Italy')
year = get_year_input()

fig_1 = plt.figure(figsize=(12, 8))
sb.set(style="white")

df_sorted_year = df_sorted_emi.sort_values(by=year, ascending=False)

sb.barplot(x=df_sorted_year[year].head(10), y=df_sorted_year['Country'].head(10),
               palette="Blues_r", edgecolor=".2")

st.pyplot(fig_1)



