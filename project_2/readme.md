![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Ames Housing Data

# Research Problem

A company in Ames, Iowa that specializes in flipping houses for resale wishes to understand the factors that affect housing prices and what features they should focus their efforts on improving to ensure the best quality of flipped houses to maximize their profits. On the other hand, they wish to also find the factors that affect the prices of old houses most negatively so that they can avoid purchasing those houses which require more work.

In other words, in the process of maximizing profits, they wish to identify which are good/bad properties, and which areas they should focus on renovating that will improve the housing prices more efficiently.

To answer their research needs, we will be using the [Ames housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) compiled by [Dean De Cock](http://jse.amstat.org/v19n3/decock.pdf), which has 80 features: 23 nominal (unordered categories), 23 ordinal (ordered categories), 14 discrete, and 20 continuous. The full explanation of the features can be found in his [data dictionary](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt). 

# Data Dictionary

|    Feature    | Type    | Description                                                    |
|:-------------:|---------|----------------------------------------------------------------|
| overall_qual  | int64   | ordinal: rates the overall material and finish of the house    |
| heating_qc    | int64   | ordinal: Heating quality and condition                         |
| gr_liv_area   | int64   | continuous: above grade (ground) living area square feet       |
| full_bath     | int64   | discrete: Full bathrooms above grade                           |
| kitchen_qual  | int64   | ordinal: Kitchen quality                                       |
| totrms_abvgrd | int64   | discrete: Total rooms above grade (does not include bathrooms) |
| fireplaces    | int64   | discrete: Number of fireplaces                                 |
| fireplace_qu  | int64   | ordinal: Fireplace quality                                     |
| lot_frontage  | float64 | continuous: Linear feet of street connected to property        |
| lot_area      | int64   | continuous: Lot size in square feet                            |
| lot_shape     | int64   | ordinal: General shape of property                             |
| exter_qual    | int64   | ordinal: Evaluates the quality of the material on the exterior |
| mas_vnr_stone | uint8   | nominal (one-hot): Masonry veneer type (stone)                 |
| mas_vnr_none  | uint8   | nominal (one-hot): Masonry veneer type (none)                  |
| mas_vnr_brick | uint8   | nominal (one-hot): Masonry veneer type (brick)                 |
| neighborhood  | int64   | ordinal: Physical locations within Ames city limits            |
| bsmt_qual     | int64   | ordinal: Evaluates the height of the basement                  |
| total_bsmt_sf | float64 | continuous: Total square feet of basement area                 |
| garage_finish | int64   | ordinal: Interior finish of the garage                         |
| garage_cars   | float64 | discrete: Size of garage in car capacity                       |
| garage_age    | float64 | discrete: age of garage                                        |
| house_age     | float64 | discrete: age of house                                         |
| remod_years   | float64 | discrete: years since remodelling                              |
| central_air   | int64   | binary: whether the house has central air-conditioning         |
| logsaleprice  | float64 | continuous: the natural logarithm of sale price of the house   |

# Modeling and Insights

Four different models were tested and compared, and the results were close:
1. Linear regression: root mean squared error (RMSE) of 23929, cross-validation score (CVS) of 31915 on train data
2. Ridge: RMSE of 24086, CSV of 31868
3. Lasso: RMSE of 24258, CSV of 31915
4. Elastic net: RMSE of 24090, CSV of 31899

As can be seen, the spread of values are small across all models, which is due to us only choosing 25 features to train our model. Because the regularization models are supposed to help with overfitting, that means that our original linear regression model was not in danger of that.

In addition, none of the models showed a significant difference between the CSV of train and test data, which means that a good balance has been reached for the bias-variance tradeoff.

Our best kaggle RMSE score was 27051, which is a decent score.

# Conclusion and Recommendations

From the selected features, it seems like attention has to be paid to the following variables:

- the material and finish of the house and its exterior quality will definitely affect the prices, and those are the more static features (harder to change)
- the gross living area, together with the sizes and quality of the garage and basement, also plays a big part
- the amenities that are important include the quality of the heating, the number and quality of fireplaces, the quality of the kitchen, the central air-conditioning, and the number of full bathrooms
- the important exterior qualities include the lot area and frontage, and the type of masonry veneer (it's better to have it than not at all!)
- the type of neighborhood also plays a huge role in the sale price, understandably
- however, the older the house and garage, and the longer since the last remodeling, the lower the sale price
- the shape of the lot also has an inverse relationship with the sale price - the more irregular it is the lower the prices!

On the topic of neighborhoods, the best seems to be Stone Brooke, Northridge Heights, Northridge, Green Hills, and Veenker, although your best bet might be Northridge Heights as it has the most number of houses but also the 2nd highest housing price.