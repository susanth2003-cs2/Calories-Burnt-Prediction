import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('imbalanced_data')

from sklearn.preprocessing import StandardScaler
import pickle

class SCALE_DATA:
    def scale(X_train, X_test):
        try:
            logger.info('Balancing data')
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
            logger.info(f'After balancing and Scaling Data : {X_train_scaled.shape}')
            logger.info(f'After balancing and Scaling Data : {X_test_scaled.shape}')
            logger.info(f'{X_train_scaled.columns}')
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            logger.info("Scaling completed successfully")

            return X_train_scaled, X_test_scaled

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
