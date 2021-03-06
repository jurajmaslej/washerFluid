# Data analysis

## Analysis process

### 1. Clean data
   1.1. Dropped all row with Target == 0 <br>
   1.2. Normalized numeric data
### 2. Check if data are consistent - plot histograms, check if data "makes sense"
   One of issues:
   Features `IsRain` and `IsDry` are not consistent. Let's explain, there are `n` days, `a` rainy days.
   One would expect that there will be approximately `n` - `a` days with tag `IsDry`.
   Histograms does not show that. Data collection/meaning needs to be checked.<br>
   ![alt text](graphs/feature_analysis/IsRain.png)
   ![alt text](graphs/feature_analysis/IsDry.png)<br>
   Other issues:<br> 2.1. Too many days with snow - approximately 1/3 of days (Explanation: data from November to February) <br>
   2.2. `TemperatureMin/Max` correlates more with `AvgVolume7D` than `AvgVolume1D`.
   Would assume that actual temperature correlates more with 1 day average than 7 day average. <br>
   2.3. Values higher than certain threshold for `Days_BA_xxxx` are grouped to one value. Not really an issue <br>
### 3. Correlation:
We are interested only in correlation with Target. We are not interested in correlation between Rain&Humidity.
Therefore check only correlation with Target (or its averages). <br>
TemperatureMax/Min has highest correlation with Target  <br>
Pressure, Humidity, WindSpeedPrecipIntCoefAvg14D, DaysBA_MgrHoliday has similar level of correlation.
Rest of the parameters has lesser influence.
   VolumeAvg features were deleted from this analysis, as their high correlation with Target is obvious.
   ![alt text](graphs/correlation/correlation_graph_Target.png)
   ![alt text](graphs/correlation/correlation_graph_sales_volume.png)
   <br> Correlation for categoric variable, correlation for volume averages can be found in `graphs/correlation`.

### 4. Principal component analysis

Dataset offers many features and it was crucial for Polynomial Regression to choose right ones.
Therefore we ran PCA algorithm. <br>
   ![alt text](graphs/feature_analysis/PCA.png)<br>

## Models:

### Linear Regression:
Start with simplest possible model first. Few Features were choosen based on their correlation.
As expected, feature with highest correlation yielded best results. However, results were unusable.
```
 TemperatureMin score: 0.09618763543946818
 TemperatureMax score: 0.10140586279345987
 Humidity       score: 0.033103791104102664
 IsRain         score: 0.06536181221743975
 IsSnow         score: 0.06886704249387932
 WindSpeed      score: 0.019518505224818572
 IsCombinedVacations score: 0.03299835518501382
 DaysBA_MigrVacation score: 0.015404750150970203
```
Score is on training data. Model was not able (as expected) to fit even the training data.

### Polynomial Regression:

Maximal number of features we were able to fit polynomial regression on was 8 due to performance limitations.
#### Polynomial regression using features highlighted by PCA
60 data points are choosen on random. Plotting all points would result in too confusing graph. <br>
Red points means prediction and target was the same. <br>
![alt text](graphs/regression/polynom_regression_unscaled_high_pca_scatter.png) <br>
#### Same features, but plotted first 100 data points <br>
![alt text](graphs/regression/polynom_regression_high_pca.png) <br>

#### Error Metric:
Results are logged to: `polynomial_regr_scores.txt`.
Best achieved results were with features : 
` ['SegmentId', 'TankAge', 'DaysBA_MigrHoliday', 'DaysBA_MigrVacation', 'DaysA_WaterChange', 'VolumeAvg1D', 'VolumeAvg3D', 'PrecipIntCoef']` <br>
Score and Errors: `score: 0.9596083458790351, mae: 0.008013301160981182, mae_on_raw_data: 57.35697880726008, % off from avg target 7.146681495956717`<br>
That means average prediction was off by **57** litters from truth. That is **7.147%** from average sold amount.

**Conclusion:** Peaks are predicted correctly, model has problem with precisely predicting target around its mean values. <br>

### SVM:

Various hyperparameters were tried. Also various features selection.<br>
However we were not able to get below `mean absolute error = 360 liters`


### Neural network

Keras with visualization in tensorboard was used. Feature selection was also applied.
Tested `keras.losses.MeanSquaredError` and `keras.losses.MeanAbsolutePercentageError`. Used `MSE`.
Computational power was limiting factor, as training had to be cut early. <br>
We experimented with various network models, none had more than 3 hidden layers. <br>
`
model = Sequential()
        model.add(Dense(12, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
`
Mae was 0.0489 which translates to around **350 liters**. That is result worse than with polynomial regression.
However, loss was still dropping and model was training, so there is further potential in using neural network.<br>
![alt text](graphs/neural_net/long_train_TensorBoard.png) <br>


### More models to implement:
Random Forest, Decision Tree


## Another approach:
Create Dataframes of data by SIDTank, train model per SIDTank. ToDo

## Time series:

### Sarima:

Results on validation data without zero-sales days<br>
![alt text](graphs_time_series/sarimax_day_of_the_week.png) <br>


Results on validation data with zero-sales days included.
From the results we can see that decision to delete 'zero-sale' days was wrong.
<br>
![alt text](graphs_time_series/sarimax_day_of_the_week_zero_sale_days_included.png) <br>

### Adding zeroes to missing dates:
Data were splitted by SIDTank and missing dates were inserted to dataframes with values set to zero <br>
Example of prediction (trained only on one SIDTank) **without zeroes** insertion:<br>
![alt text](graphs_time_series/sarimax_on_tank_ids/110706-7.png) <br>
Example of prediction (trained only on one SIDTank) **with zeroes** insertion:<br>
![alt text](graphs_time_series/sarimax_dates_filled_zeroes_on_tank_ids/110706-7.png) <br>
We can see **better** results for prediction of low/zero sales


### Group by Date on Locations:
Dataframe created from tanks from same Location ('SIDTank' without ```-x```).
This group by got us multiple entries for one day. We used group by with ```mean()```.
Results were not much better.
This might indicate there is not direct and strong connection between sales on various tanks on same Location.<br>
Or model was not able to fit even this data <br>
![alt text](graphs_time_series/Sarimax_dates_filled_zeroes_on_locations_groupby_date/110706-7.png) <br>

## Random Forest:
ToDo
