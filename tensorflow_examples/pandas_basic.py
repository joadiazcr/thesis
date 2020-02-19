import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print "Pandas version: %s" %pd.__version__

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print "\n %s" %cities
print(type(cities['City name']))
print cities['City name']
print(type(cities['City name'][1]))
print cities['City name'][1]
print(type(cities[0:2]))
print cities[0:2]
print population / 1000.
print np.log(population)
population.apply(lambda val: val > 1000000)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['Population density'] = cities['Population'] / cities['Area square miles']
print "\n %s" %cities

cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
print "\n %s" %cities

print "\n %s" %city_names.index
print "\n %s" %cities.index

cities = cities.reindex([2, 0, 1])
print "\n %s" %cities
cities = cities.reindex(np.random.permutation(cities.index))
print "\n %s" %cities


california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
print "\n %s" %california_housing_dataframe.head()
print "\n %s" %california_housing_dataframe.describe()
california_housing_dataframe.hist('housing_median_age')
plt.show()
