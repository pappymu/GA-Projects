![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Standardized Test Analysis

## Overview

There have been claims that standardized testing propagate economic inequality, for reasons such as better access to preparation, including a 2013 paper that claims "income is linearly related to SAT performance". There was also an article by the National Education Association that links the racist beginnings of standardized testing with eugenics. There was even a lawsuit in 2019 claiming that the SAT and ACT are racially and economically biased, but some claim other factors like lack of test preparation and bribery.

A College Board (administrator of the SAT) research report claimed that "the vast majority of the SAT–grade relationship is independent of SES \[socio-economic status\]", while another 2013 College Board report shows a proportional relationship between family income and scores attained. The ACT, on the other hand, published a 2016 report shows a flat relationship between family income and the scores attained.

In the face of seemingly conflicting information, it would be prudent to perform an analysis on test results for both standardized tests, comparing the scores attained and participation rates with the economic status across states.

## Problem Statement

How much does the economic background of students in US states affect their ACT/SAT scores and participation rates?

---

## Datasets Chosen

Of the datasets provided, we chose 2 to work with:

* [`act_2019.csv`](./data/act_2019.csv): 2019 ACT Scores by State
* [`sat_2019.csv`](./data/sat_2019.csv): 2019 SAT Scores by State

The first is a list of (average) ACT composite scores and participation rate by state, and the second is a list of (average) total SAT scores and participation rate by state.

In addition, we also examine a [list of US states by GDP](https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP_per_capita), which provides the GDP per capita for each US state in 2019, exactly the time period we will analyse.

## Data Dictionary

| Feature        | Type   | Dataset   | Description                                                                       |
| ---            | ---    | ---       | ---                                                                               |
| state          | object | act_2019  | The names of the 52 states in the US (not including territories and dependencies) |
| state_abbr     | object | -         | The abbreviated names of the states for easier plotting.                          |
| act            | float  | act_2019  | average ACT composite score by state                                              |
| act_part_rate  | float  | act_2019  | ACT participation rate by state                                                   |
| sat            | int    | sat_2019  | average SAT composite score by state                                              |
| sat_part_rate  | float  | sat_2019  | SAT participation rate by state                                                   |
| gdp_per_capita | float  | state_gdp | The GDP per capita of each US state in 2019, taken from wikipedia.                |

---

## Summary of Analysis

From the high-level statistics, a few patterns emerged:
- participation rates for both tests have a large spread, as evidenced by the standard deviations being almost the same as the mean
- ACT participation rate is smaller than the SAT one
- test scores have relatively small standard deviations

Four different types of visualisations were then plotted:
- Heatmap:
    - GDP has a low correlation (0.38 with ACT and -0.21 with SAT) with the other variables
    - both tests have a strong inverse relationship between the grades and the participation rates (-0.87 for ACT and -0.86 for SAT)
    - ACT and SAT participation rates also have a strong inverse correlation (-0.87)
- Histogram:
    - none of the data follow a normal curve
    - could be due to lack of data points
- Boxplot:
    - large interquartile range of the participation rates indicate large spread
    - mean of scores skewed to the lower end
    - large skew to the lower income
- Scatterplot
    - weak correlation between GDP against scores or participation rates
    - inverse trend between ACT than SAT takers
    - inverse relationship between scores and test participation rates

## Conclusion and Recommendations

Throughout our analysis, a few points have been salient:
- there is low correlation between the GDP per capita of the states and their scores in the standardized tests (p=0.38 for ACT and p=-0.21 for SAT)
- an inverse relationship is prominent between the scores and the participation rates, i.e. the less students taking the tests, the higher the average grades (p=-0.87 for ACT, p=-0.86 for SAT)
- there is also an inverse relationship between the participation rates for both tests, i.e. students who take the SAT generally do not take the ACT and vice versa (p=-0.87)
- there is a significant number of states with low SAT participation, compared to the ACT

Recommendations for future work:
- GDP per capita is not the only economic metric that should be considered, others include household income, personal income, family income etc. 
- other factors that affect academic performance should also be considered, e.g. access to resources, teaching quality scores
- more data could also be gathered to see the trends across various years, which might allow clearer patterns to emerge.
- consider why participation rates vary across states