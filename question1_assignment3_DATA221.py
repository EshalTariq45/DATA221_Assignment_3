# question 1
import pandas as pd

dataframe_for_crime=pd.read_csv("C:/Users/et827/Downloads/crime1.csv")
#putting csv crime file into a dataframe

violent_crimes_per_pop=dataframe_for_crime["ViolentCrimesPerPop"]
#focusing on the column ViolentCrimesPerPop

mean_value=violent_crimes_per_pop.mean()
median_value=violent_crimes_per_pop.median()
standard_deviation_value=violent_crimes_per_pop.std()
minimum_value=violent_crimes_per_pop.min()
maximum_value=violent_crimes_per_pop.max()
#calculating the values and storing them to print out

print("the mean value of the crime file: ", mean_value)
print("the median value of the crime file: ",median_value)
print("the standard deviation value of the crime file: ",standard_deviation_value)
print("the minimum value of the crime file: ", minimum_value)
print("the maximum value of the crime file: ", maximum_value)

# COMPARING MEAN AND MEDIAN:
#when printed, the mean value is 0.44, and the median value is 0.39. Since mean> median,
#       it means that the distribution is right-skewed. There are outliers in the data
#       with higher values that in turn pull the mean upward

# EXTREME VALUE QUESTION:
#if there are extreme values (very large or very small), the mean is more affected. This
#       is because the mean uses every value in the data set when being calculated. The
#       formula is : sum of all values/ number of all values, showing that the mean will
#       be more affected than the median