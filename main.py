import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns

import warnings

from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from random_sample import RSITechnique
from var_out import VT_OUT
from feature_selection import FEATURE_SELECTION
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imbalanced_data import SCALE_DATA
from models import LR_MODEL

class CALORIES_BURNT:
    def __init__(self,calories_path,exercise_path):
        try:
            self.calories_df = pd.read_csv(calories_path)
            self.exercise_df = pd.read_csv(exercise_path)

            logger.info("Both CSV files loaded successfully")
            logger.info(f'{self.calories_df.sample(10)}')
            logger.info(f'{self.exercise_df.sample(10)}')

            self.df = pd.merge(self.exercise_df,self.calories_df,on='User_ID',how='inner')
            self.df = self.df.drop(columns=['User_ID'])
            logger.info(f'After merging{self.df.sample(10)}')
            logger.info(f'{self.df.shape}')
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f'{self.df.isnull().sum()}')

            #Independent and Dependent variables
            self.X = self.df.iloc[: , :-1]
            self.y = self.df.iloc[: , -1]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state = 42)

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            logger.info(f'Missing Values')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            if self.X_train.isnull().sum().any() > 0 or self.X_test.isnull().sum().any() > 0:
                self.X_train, self.X_test = RSITechnique.random_sample_imputation_technique(self.X_train, self.X_test)
            else:
                logger.info(f'No Missing Values')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def VarTrasform_Outliers(self):
        try:
            logger.info(f'Variable Transform Outliers Columns')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            self.X_train_num = self.X_train.select_dtypes(exclude = 'object')
            self.X_train_cat = self.X_train.select_dtypes(include = 'object')
            self.X_test_num = self.X_test.select_dtypes(exclude = 'object')
            self.X_test_cat = self.X_test.select_dtypes(include = 'object')

            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')

            self.X_train_num, self.X_test_num = VT_OUT.variable_transformation_outliers(self.X_train_num, self.X_test_num)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    '''
    def fs(self):
        try:
            logger.info(f'Feature Selection')
            logger.info(f" Before : {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"Before : {self.X_test_num.columns} -> {self.X_test_num.shape}")

            self.X_train_num, self.X_test_num = FEATURE_SELECTION.complete_feature_selection(self.X_train_num, self.X_test_num, self.y_train)

            logger.info(f" After : {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"After : {self.X_test_num.columns} -> {self.X_test_num.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    '''
    def cat_to_num(self):
        try:
            logger.info('Categorical to Numerical')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            for i in self.X_train_cat.columns:
                logger.info(f'{i} --> {self.X_train_cat[i].unique()}')

            logger.info(f'Before Converting : {self.X_train_cat}')
            logger.info(f'Before Converting : {self.X_test_cat}')

            #One-Hot Encoding
            one_hot = OneHotEncoder(drop = 'first')
            one_hot.fit(self.X_train_cat[['Gender']])
            res = one_hot.transform(self.X_train_cat[['Gender']]).toarray()

            f = pd.DataFrame(data = res, columns = one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop = True, inplace = True)
            f.reset_index(drop = True, inplace = True)

            self.X_train_cat = pd.concat([self.X_train_cat, f],axis = 1)
            self.X_train_cat = self.X_train_cat.drop(['Gender'], axis = 1)

            res1 = one_hot.transform(self.X_test_cat[['Gender']]).toarray()
            f1 = pd.DataFrame(data = res1, columns = one_hot.get_feature_names_out())

            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)

            self.X_test_cat = pd.concat([self.X_test_cat, f1],axis = 1)
            self.X_test_cat = self.X_test_cat.drop(['Gender'], axis = 1)

            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f"After Converting : {self.X_train_cat}")
            logger.info(f"After Converting : {self.X_test_cat}")

            logger.info(f"{self.X_train_cat.shape}")
            logger.info(f"{self.X_test_cat.shape}")

            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")

            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f"{self.training_data.shape}")
            logger.info(f"{self.testing_data.shape}")

            logger.info(f"{self.training_data.isnull().sum()}")
            logger.info(f"{self.testing_data.isnull().sum()}")

            logger.info(f"=======================================================")

            logger.info(f"Training Data : {self.training_data.sample(10)}")
            logger.info(f"Testing Data : {self.testing_data.sample(10)}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def data_balance(self):
        try:
            logger.info('Scaling Data Before Regression')
            self.X_train_num, self.X_test_num = SCALE_DATA.scale(self.X_train_num, self.X_test_num)

            LR_MODEL.linear_regression(self.X_train_num, self.y_train, self.X_test_num, self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

if __name__ == "__main__":
    try:
        obj = CALORIES_BURNT('C:\\Users\\Rajesh\\Downloads\\Mini Projects\\Calories Burnt Prediction\\calories (1).csv','C:\\Users\\Rajesh\\Downloads\\Mini Projects\\Calories Burnt Prediction\\exercise.csv')
        obj.missing_values()
        obj.VarTrasform_Outliers()
        #obj.fs()
        obj.cat_to_num()
        obj.data_balance()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
