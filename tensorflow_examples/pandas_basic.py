import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print "Pandas version: %s" %pd.__version__

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print "\n %s" %cities
print "\n City name type: %s" % (type(cities['City name']))
print cities['City name']
print "\n Type of each element of City name: %s" % (type(cities['City name'][1]))
print cities['City name'][1]
print "\n Type of cites[0:2]: %s" % (type(cities[0:2]))
print cities[0:2]
print "\nPopulation/1000"
print population / 1000.0
print "\nlog(population)"
print np.log(population)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
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
print "\nCalifornia housing data frame head"
print california_housing_dataframe.head()
print "\nCalifornia housing data frame describe"
print california_housing_dataframe.describe()

california_housing_dataframe.hist('housing_median_age')
plt.show()
