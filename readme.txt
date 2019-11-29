# MSBD5001 Personal Project: Kaggle in-class competition
## Introduction
1. The programming language used  in this project is python3.5
2. The packages used in this project including: pandas, numpy, sklearn, scipy, xgboost, matplotlib, seaborn
## Project Details
### Input
Input files are stored under `rawdata` dolder, including `samplesubmission.csv` `test.csv` and `train.csv`.
### Data processing and Feature Engineering
To run this project, firstly you have to perform data pre-process and feature engineering.
1. Under `dataprocessiing` folder, run `process.py`, perform data pre-process, you can get two CSV files named `prefeatures_dropold.csv` and `test_feature.csv` under `dataprocessiing/processed data` folder.
2. Run `feature ranking.py` under `dataprocessiing/FeatureEngineering` folder, you can get a CSV file named `slctdfeature .csv` under `dataprocessiing/processed data` folder. But because of the randomness of running result of randomness `feature ranking.py`, if you want to generate my final submission, I have uploaded the `slctdfeature .csv` file that I used in my prediction models under `dataprocessiing/processed data` folder, plesase use this file directly.
3. Run `KFold.py`, it will show cross validation result of several diffrent regression models.
### Random Forest Model
Run `randomforest_selctdft.py` under `submit1` folder, it can generate the resulting test.csv file under `submit1` folder.
### XGBoost Model
Run `xgboost_selctdft.py` under `submit2` folder, it can generate the resulting test.csv file under `submit1` folder.
### Results
The resulting `test.csv ` file I submitted on kaggle website are stored under `submit1` folder and `submit2` folder.
