```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
```

## Predicting College Major Enrollment

### Stat 426 Final Project

A few years ago, Five Thirty Eight published an article about the salaries of various college majors. The article explored which degrees could be the most profitable for students. This made me curious: do students actually flock to higher paying degrees? From what I've observed in my college experience, many male BYU students do choose higher paying career paths because they are concerned about providing for a family. Yet many other college students around the country complain that they are unable to pay off their student loans— is this the result of many students choosing lower paying jobs? I decided to analyze the same dataset to see if economic factors such as wages and unemployment rates did have an effect on how many people enrolled in a particular program. Do more students choose higher paying jobs or lower paying jobs? The data I am exploring from Five Thirty Eight is originally from the American Community Survey 2010-2012 Public Use Microdata Series.


```python
#Reading in the data
table2 = pd.read_html("https://github.com/fivethirtyeight/data/blob/master/college-majors/grad-students.csv")
```


```python
gradstudent = table2[0]
gradstudent.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Major_code</th>
      <th>Major</th>
      <th>Major_category</th>
      <th>Grad_total</th>
      <th>Grad_sample_size</th>
      <th>Grad_employed</th>
      <th>Grad_full_time_year_round</th>
      <th>Grad_unemployed</th>
      <th>Grad_unemployment_rate</th>
      <th>...</th>
      <th>Nongrad_total</th>
      <th>Nongrad_employed</th>
      <th>Nongrad_full_time_year_round</th>
      <th>Nongrad_unemployed</th>
      <th>Nongrad_unemployment_rate</th>
      <th>Nongrad_median</th>
      <th>Nongrad_P25</th>
      <th>Nongrad_P75</th>
      <th>Grad_share</th>
      <th>Grad_premium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>5601</td>
      <td>CONSTRUCTION SERVICES</td>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>9173</td>
      <td>200</td>
      <td>7098</td>
      <td>6511</td>
      <td>681</td>
      <td>0.087543</td>
      <td>...</td>
      <td>86062</td>
      <td>73607</td>
      <td>62435</td>
      <td>3928</td>
      <td>0.050661</td>
      <td>65000.0</td>
      <td>47000</td>
      <td>98000.0</td>
      <td>0.096320</td>
      <td>0.153846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>6004</td>
      <td>COMMERCIAL ART AND GRAPHIC DESIGN</td>
      <td>Arts</td>
      <td>53864</td>
      <td>882</td>
      <td>40492</td>
      <td>29553</td>
      <td>2482</td>
      <td>0.057756</td>
      <td>...</td>
      <td>461977</td>
      <td>347166</td>
      <td>250596</td>
      <td>25484</td>
      <td>0.068386</td>
      <td>48000.0</td>
      <td>34000</td>
      <td>71000.0</td>
      <td>0.104420</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>6211</td>
      <td>HOSPITALITY MANAGEMENT</td>
      <td>Business</td>
      <td>24417</td>
      <td>437</td>
      <td>18368</td>
      <td>14784</td>
      <td>1465</td>
      <td>0.073867</td>
      <td>...</td>
      <td>179335</td>
      <td>145597</td>
      <td>113579</td>
      <td>7409</td>
      <td>0.048423</td>
      <td>50000.0</td>
      <td>35000</td>
      <td>75000.0</td>
      <td>0.119837</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>2201</td>
      <td>COSMETOLOGY SERVICES AND CULINARY ARTS</td>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>5411</td>
      <td>72</td>
      <td>3590</td>
      <td>2701</td>
      <td>316</td>
      <td>0.080901</td>
      <td>...</td>
      <td>37575</td>
      <td>29738</td>
      <td>23249</td>
      <td>1661</td>
      <td>0.052900</td>
      <td>41600.0</td>
      <td>29000</td>
      <td>60000.0</td>
      <td>0.125878</td>
      <td>0.129808</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>2001</td>
      <td>COMMUNICATION TECHNOLOGIES</td>
      <td>Computers &amp; Mathematics</td>
      <td>9109</td>
      <td>171</td>
      <td>7512</td>
      <td>5622</td>
      <td>466</td>
      <td>0.058411</td>
      <td>...</td>
      <td>53819</td>
      <td>43163</td>
      <td>34231</td>
      <td>3389</td>
      <td>0.072800</td>
      <td>52000.0</td>
      <td>36000</td>
      <td>78000.0</td>
      <td>0.144753</td>
      <td>0.096154</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python

```

### Data Cleaning

To clean the data, I am going to be dropping some unnecessary columns. I am also going to rearrange the table and create a variable that indicates what type of degree the student is getting (graduate or undergraduate).


```python
#Drop the Unnamed: 0 column, which has no useful data
gradstudent.drop('Unnamed: 0', inplace=True, axis=1)
```


```python
justgrad = gradstudent[['Major', 'Major_category', 'Grad_total', 'Grad_employed', 'Grad_unemployed', 'Grad_unemployment_rate', 'Grad_median', 'Grad_P25', 'Grad_P75']]
justnongrad = gradstudent[['Major', 'Major_category', 'Nongrad_total', 'Nongrad_employed', 'Nongrad_unemployed', 'Nongrad_unemployment_rate', 'Nongrad_median', 'Nongrad_P25', 'Nongrad_P75']]
```


```python
#renaming columns
justnongrad = justnongrad.rename(columns={"Nongrad_total": "Totalnum", "Nongrad_employed": "Employed", "Nongrad_unemployed": "Unemployed", "Nongrad_unemployment_rate": "Unemployment_rate", "Nongrad_median": "Median_wage", "Nongrad_P25": "Q25_wage", "Nongrad_P75": "Q75_wage"})
justgrad = justgrad.rename(columns={"Grad_total": "Totalnum", "Grad_employed": "Employed", "Grad_unemployed": "Unemployed", "Grad_unemployment_rate": "Unemployment_rate", "Grad_median": "Median_wage", "Grad_P25": "Q25_wage", "Grad_P75": "Q75_wage"})
```


```python
#creating degree variable
justgrad['degree'] = "grad"
justnongrad['degree'] = "undergrad"
```


```python
#combining tables, setting index
college = justnongrad.append(justgrad)
college = college.reset_index().drop('index', axis=1)
college.set_index('Major', inplace=True)
```


```python
#Dropping some more variables that are redundant
college.drop('Employed', axis=1, inplace=True)
college.drop('Unemployed', axis=1, inplace=True)
```


```python
college.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 346 entries, CONSTRUCTION SERVICES to EDUCATIONAL ADMINISTRATION AND SUPERVISION
    Data columns (total 7 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Major_category     346 non-null    object 
     1   Totalnum           346 non-null    int64  
     2   Unemployment_rate  346 non-null    float64
     3   Median_wage        346 non-null    float64
     4   Q25_wage           346 non-null    int64  
     5   Q75_wage           346 non-null    float64
     6   degree             346 non-null    object 
    dtypes: float64(3), int64(2), object(2)
    memory usage: 21.6+ KB
    


```python
college
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Major_category</th>
      <th>Totalnum</th>
      <th>Unemployment_rate</th>
      <th>Median_wage</th>
      <th>Q25_wage</th>
      <th>Q75_wage</th>
      <th>degree</th>
    </tr>
    <tr>
      <th>Major</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CONSTRUCTION SERVICES</th>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>86062</td>
      <td>0.050661</td>
      <td>65000.0</td>
      <td>47000</td>
      <td>98000.0</td>
      <td>undergrad</td>
    </tr>
    <tr>
      <th>COMMERCIAL ART AND GRAPHIC DESIGN</th>
      <td>Arts</td>
      <td>461977</td>
      <td>0.068386</td>
      <td>48000.0</td>
      <td>34000</td>
      <td>71000.0</td>
      <td>undergrad</td>
    </tr>
    <tr>
      <th>HOSPITALITY MANAGEMENT</th>
      <td>Business</td>
      <td>179335</td>
      <td>0.048423</td>
      <td>50000.0</td>
      <td>35000</td>
      <td>75000.0</td>
      <td>undergrad</td>
    </tr>
    <tr>
      <th>COSMETOLOGY SERVICES AND CULINARY ARTS</th>
      <td>Industrial Arts &amp; Consumer Services</td>
      <td>37575</td>
      <td>0.052900</td>
      <td>41600.0</td>
      <td>29000</td>
      <td>60000.0</td>
      <td>undergrad</td>
    </tr>
    <tr>
      <th>COMMUNICATION TECHNOLOGIES</th>
      <td>Computers &amp; Mathematics</td>
      <td>53819</td>
      <td>0.072800</td>
      <td>52000.0</td>
      <td>36000</td>
      <td>78000.0</td>
      <td>undergrad</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>COUNSELING PSYCHOLOGY</th>
      <td>Psychology &amp; Social Work</td>
      <td>51812</td>
      <td>0.035600</td>
      <td>50000.0</td>
      <td>36000</td>
      <td>65000.0</td>
      <td>grad</td>
    </tr>
    <tr>
      <th>CLINICAL PSYCHOLOGY</th>
      <td>Psychology &amp; Social Work</td>
      <td>22716</td>
      <td>0.044958</td>
      <td>70000.0</td>
      <td>47000</td>
      <td>95000.0</td>
      <td>grad</td>
    </tr>
    <tr>
      <th>HEALTH AND MEDICAL PREPARATORY PROGRAMS</th>
      <td>Health</td>
      <td>114971</td>
      <td>0.021687</td>
      <td>135000.0</td>
      <td>70000</td>
      <td>294000.0</td>
      <td>grad</td>
    </tr>
    <tr>
      <th>SCHOOL STUDENT COUNSELING</th>
      <td>Education</td>
      <td>19841</td>
      <td>0.051400</td>
      <td>56000.0</td>
      <td>42000</td>
      <td>70000.0</td>
      <td>grad</td>
    </tr>
    <tr>
      <th>EDUCATIONAL ADMINISTRATION AND SUPERVISION</th>
      <td>Education</td>
      <td>54159</td>
      <td>0.016761</td>
      <td>65000.0</td>
      <td>50000</td>
      <td>86000.0</td>
      <td>grad</td>
    </tr>
  </tbody>
</table>
<p>346 rows × 7 columns</p>
</div>




```python

```

### Exploratory Data Analysis

To explore the data, I am going to look at some plots and summary statistics to get an idea of what the data look like.


```python
#Pairplot of the variables
import seaborn as sns
sns.pairplot(college)
```




    <seaborn.axisgrid.PairGrid at 0x15467e180d0>




    
![png](output_20_1.png)
    


It looks like some of the variables are correlated. All 3 wage metrics are correlated, so I will try dropping two of them when looking at feature importance.


```python
#plot of wages and enrollment
plt.scatter(college['Median_wage'], college['Totalnum'])
plt.xlabel("Median Wage")
plt.ylabel("Enrollment (in millions)")
plt.title("Median Wage vs Enrollment")
```




    Text(0.5, 1.0, 'Median Wage vs Enrollment')




    
![png](output_22_1.png)
    



```python
#plot of unemployment and enrollment
plt.scatter(college['Unemployment_rate'], college['Totalnum'])
plt.xlabel("Unemployment Rate")
plt.ylabel("Enrollment (in millions)")
plt.title("Unemployment Rate vs Enrollment")
```




    Text(0.5, 1.0, 'Unemployment Rate vs Enrollment')




    
![png](output_23_1.png)
    


An initial look at a plot of median wage and student enrollment does not show any obvious trends. Most majors seems to have below average enrollment numbers. The points look more densely populated around low salary and low enrollment than around high salary and low enrollment.


```python
#Means and Medians for different variables
print(college.mean())
print(college.median())
print(college[college['degree'] == 'grad'].mean())
print(college[college['degree'] == 'grad'].median())
print(college[college['degree'] == 'undergrad'].mean())
print(college[college['degree'] == 'undergrad'].median())
```

    Totalnum             171196.167630
    Unemployment_rate         0.046645
    Median_wage           67669.797688
    Q25_wage              46337.343931
    Q75_wage              98210.144509
    dtype: float64
    Totalnum             54011.500000
    Unemployment_rate        0.044623
    Median_wage          65000.000000
    Q25_wage             45000.000000
    Q75_wage             95000.000000
    dtype: float64
    Totalnum             127672.023121
    Unemployment_rate         0.039343
    Median_wage           76755.780347
    Q25_wage              52596.508671
    Q75_wage             112087.341040
    dtype: float64
    Totalnum              37872.000000
    Unemployment_rate         0.036654
    Median_wage           75000.000000
    Q25_wage              50000.000000
    Q75_wage             108000.000000
    dtype: float64
    Totalnum             214720.312139
    Unemployment_rate         0.053947
    Median_wage           58583.815029
    Q25_wage              40078.179191
    Q75_wage              84332.947977
    dtype: float64
    Totalnum             68993.000000
    Unemployment_rate        0.051031
    Median_wage          55000.000000
    Q25_wage             38000.000000
    Q75_wage             80000.000000
    dtype: float64
    


```python
college['Totalnum'].sort_values()
```




    Major
    COURT REPORTING                                                         1542
    MISCELLANEOUS FINE ARTS                                                 1733
    SCHOOL STUDENT COUNSELING                                               2232
    ACTUARIAL SCIENCE                                                       2472
    ELECTRICAL, MECHANICAL, AND PRECISION TECHNOLOGIES AND PRODUCTION       3187
                                                                          ...   
    GENERAL EDUCATION                                                    1388324
    NURSING                                                              1686899
    ACCOUNTING                                                           1708418
    GENERAL BUSINESS                                                     2056867
    BUSINESS MANAGEMENT AND ADMINISTRATION                               2996892
    Name: Totalnum, Length: 346, dtype: int64



The mean number of graduate students enrolled is 127672, but the median is only 37872. Likewise, the mean number of undergraduate students is 214720, while the median is only 68993. The few majors with extremely large enrollment, like the undergraduate program in Business Management and Administration, are skewing the averages.


```python
print("Minimum number of people in a college program:", college['Totalnum'].min())
print("Maximum number of people in a college program:", college['Totalnum'].max())
print("Range:", college['Totalnum'].max() - college['Totalnum'].min())
```

    Minimum number of people in a college program: 1542
    Maximum number of people in a college program: 2996892
    Range: 2995350
    


```python

```

### Models

I am going to try a variety of models to see which one can best predict how many students are enrolled in a major. First, I am going to do a basic Linear Regression. Linear Regression is a great first look at the data. I am also going to look at Lasso and Ridge regression because the two methods adjust the coefficients for the variables so the more important variables are contributing more to the final predictions. Lastly, I am going to look at K Neighbors and Decision Trees. I will compare the root mean square error in each model to find the one with the least error.


```python
#Separating response variable from the features
X = pd.concat([college.iloc[:,0:1], college.iloc[:,2:]], axis=1)
y = college['Totalnum']
```


```python
#Encoding the categorical variables as dummy variables so we can use them in our regression models
import category_encoders as ce
encoder = ce.OneHotEncoder(use_cat_names=True)
X_encoded = encoder.fit_transform(X)
```

    C:\Users\rhunter\anaconda3\lib\site-packages\category_encoders\utils.py:21: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      elif pd.api.types.is_categorical(cols):
    


```python
#Split the data into testing and training sets
#I am stratifying based on type of degree, because there are generally fewer graduate students
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state = 385, test_size = 0.25, stratify=X_encoded['degree_grad'])
```

Linear Regression


```python
lm = LinearRegression()
lm.fit(X_train, y_train)
yhat_lm = lm.predict(X_test)
mse_lm = mean_squared_error(y_test, yhat_lm)
print("Root Mean Squared Error:", np.sqrt(mse_lm))
print("Mean Absolute Error:", mean_absolute_error(y_test, yhat_lm))
```

    Root Mean Squared Error: 242431.45524504167
    Mean Absolute Error: 176302.66780445795
    

Lasso Regression:
Lasso can make the coefficients of unimportant factors 0, but it may not work with all types of data.


```python
lass = Lasso(alpha=3)
lass.fit(X_train, y_train)
yhat_lass = lass.predict(X_test)
mse_lass = mean_squared_error(y_test, yhat_lass)
print("Root Mean Squared Error:", np.sqrt(mse_lass))
print("Mean Absolute Error:", mean_absolute_error(y_test, yhat_lass))
```

    Root Mean Squared Error: 242443.4290509495
    Mean Absolute Error: 176300.09494422065
    

Ridge Regression:
Ridge can make the coefficients of unimportant factors very small, but it may struggle with certain types of data.


```python
ridg = Ridge(alpha=.0001)
ridg.fit(X_train, y_train)
yhat_ridg = ridg.predict(X_test)
mse_ridg = mean_squared_error(y_test, yhat_ridg)
print("Root Mean Squared Error:", np.sqrt(mse_ridg))
print("Mean Absolute Error:", mean_absolute_error(y_test, yhat_ridg))
```

    Root Mean Squared Error: 242436.4941595858
    Mean Absolute Error: 176301.95410764325
    

K Neighbors Regression: K neighbors looks at data points with similar features to figure out predictions. It is often quite accurate, but might not perform well on our outliers.


```python
#Find best parameter
from sklearn.model_selection import GridSearchCV
parameters_knn = {'n_neighbors':list(range(1,30))}
knncv = KNeighborsRegressor()
parknncv = GridSearchCV(knncv, parameters_knn)
parknncv.fit(X_train, y_train)
parknncv.best_params_
```




    {'n_neighbors': 28}




```python
knn = KNeighborsRegressor(n_neighbors=28)
knn.fit(X_train, y_train)
yhat_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, yhat_knn)
print("Root Mean Squared Error:", np.sqrt(mse_knn))
print("Mean Absolute Error:", mean_absolute_error(y_test, yhat_knn))
```

    Root Mean Squared Error: 208322.0840649877
    Mean Absolute Error: 159680.66666666672
    

Decision Tree Regression: 
Decision trees are very intuitive to understand, but they can also be very sensitive to small changes in data.


```python
tree = DecisionTreeRegressor(min_samples_leaf=3)
tree.fit(X_train, y_train)
yhat_tree = tree.predict(X_test)
mse_tree = mean_squared_error(y_test, yhat_tree)
print("Root Mean Squared Error:", np.sqrt(mse_tree))
print("Mean Absolute Error:", mean_absolute_error(y_test, yhat_tree))
```

    Root Mean Squared Error: 328364.45124334656
    Mean Absolute Error: 187546.50957854406
    

The best model out of these 5 is a K Neighbors Regressor. It has a root mean squared error of 208322 and a mean absolute error of 159680. Both of these measurements were the lower compared to the corresponding measurement of the other models. Although it is the most accurate, the root mean squared error is still pretty big, especially considering that the average number of students enrolled in a major was 171196. There is a very large range in our data, spanning 2995350.

I used GridSearchCV to choose the best hyperparameters for my models. For my K Neighbors Regressor, the best hyperparameter was 28.


```python
#Evaluating Model Performance
#Cross Validated Model Performance
scores_knn = cross_val_score(knn, X_encoded, y, cv=10, scoring='neg_mean_squared_error')
mse_knn_cv = -1*scores_knn.mean()
print("CV Root Mean Squared Error:", np.sqrt(mse_knn_cv))
```

    CV Root Mean Squared Error: 333261.4062367283
    

In cross validation, my model did not perform very well. The average root mean squared error of my cross validation trials was 333261, which is much bigger than the root mean squared error from my model's training set. Because this model has such high errors, I'm not confident that it can accurately predict how many students are enrolled in a major based on economic information.


```python
#Looking at coefficients for the reatures to see relative importance
#removing percentiles for wage and just looking at median wage
x = X_train.drop('Q25_wage', axis=1)
x.drop('Q75_wage', axis=1, inplace=True)
lmx = LinearRegression()
lmx.fit(ex, y_train)
feature_names = list(ex.columns.values)
list(zip(lmx.coef_, feature_names))
```




    [(-54045.22717077238, 'Major_category_Industrial Arts & Consumer Services'),
     (-11341.243751724425, 'Major_category_Arts'),
     (429414.6335566854, 'Major_category_Business'),
     (-46211.12199257608, 'Major_category_Computers & Mathematics'),
     (-47535.08739525528, 'Major_category_Law & Public Policy'),
     (-173249.0285572183, 'Major_category_Agriculture & Natural Resources'),
     (97755.0845611379, 'Major_category_Communications & Journalism'),
     (-131953.87011979855, 'Major_category_Engineering'),
     (107773.46753205455, 'Major_category_Social Science'),
     (-5065.382069103262, 'Major_category_Health'),
     (-146836.136242335, 'Major_category_Interdisciplinary'),
     (-119339.30108950294, 'Major_category_Physical Sciences'),
     (41614.259334271395, 'Major_category_Humanities & Liberal Arts'),
     (69993.91149008226, 'Major_category_Psychology & Social Work'),
     (-100565.18406684167, 'Major_category_Biology & Life Science'),
     (89590.22603794931, 'Major_category_Education'),
     (-2168137.9954188573, 'Unemployment_rate'),
     (0.8659846057489631, 'Median_wage'),
     (65603.30823899244, 'degree_undergrad'),
     (-65603.30823900696, 'degree_grad')]



Based on these coefficients from the basic linear model, wages do have an effect on student enrollment. A higher median wage means more people choose that major. Unemployment rate also impacts the number of students enrolled in a major; generally as unemployment rate goes up, the number of students enrolled in the major goes down. This suggests that people gravitate more toward degrees with lower unemployment rates.


```python


```

### Conclusions

Economic factors can help us predict how many students are enrolled in a major or program. If a college major had a greater median wage, more people chose that program. At the same time, a lower unemployment rate was also associated with more students enrolling in that major. This suggests that students are indeed choosing majors that will be good for them economically. 

This data is a few years old, and there are lots of other variables that could have been included in an analysis like this. I would be interested in trying to answer the same question with a bigger dataset.


```python

```
