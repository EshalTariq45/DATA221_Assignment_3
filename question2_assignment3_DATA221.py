#question 2
import pandas as pd
import matplotlib.pyplot as plt


dataframe_for_crime=pd.read_csv("C:/Users/et827/Downloads/crime1.csv")
#putting csv crime file into a dataframe

violent_crimes_per_pop=dataframe_for_crime["ViolentCrimesPerPop"]
#focusing on the column ViolentCrimesPerPop


plt.hist(violent_crimes_per_pop,bins=20, edgecolor="black")
plt.title("Histogram Distribution of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.show()
#plotting histogram, along with the title, x label and y label

#explanation for histogram-
#the histogram above shows that most of the values are concentrated between 0.1 and 0.7.
#   the distribution appears to be slightly positively skewed, or right skewed, as there
#   are values that extend further to the right
#   this shows that while most communities do have lower-moderate violent crimes, there is a
#   smaller number that has relatively higher rates.



plt.boxplot(violent_crimes_per_pop)
plt.title("Boxplot of distribution of violent crimes per population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Values")
plt.show()
#plotting boxplot, along with the title, x label and y label

#explanation for boxplot-
#the box plot above shows that the median is slightly below the center of the box, which also
#   supports the histogram being skewed to the right. The upper whiskers go further than the
#   lower whisker, which shows that there may be extreme high values present due to
#   a longer tail on the higher end

