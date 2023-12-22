# python version 3.11.3

import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)

import pickle

#----------------------------------------------------------------------------------

def read_data(path):
    """Reads a CSV file and returns the data as a pandas dataframe."""
    df = pd.read_csv(path)
    return df

#----------------------------------------------------------------------------------

def prevent_data_leak(df):
    """- Divides the original dataset into train and validation sets to validate the model and prevent data leakage.
    - Saves the split dataset into CSV format."""
    train, validation = train_test_split(df, test_size=0.2, random_state=42)
    validation.to_csv("house_price_validation.csv", index=False)
    train.reset_index(drop=True, inplace=True)
    return train

#----------------------------------------------------------------------------------

def drop_id(df):
    """Removes the alphanumeric feature 'Id' from the dataset."""
    df.drop(columns='Id', inplace=True)
    return df

#----------------------------------------------------------------------------------

def drop_outliers(df):
    """Eliminates outliers based on specific conditions:
    - Properties with 'GrLivArea' greater than 4,000 sqft.
    - 'SalePrice' exceeding 700,000.
    - 'LotArea' surpassing 100,000 sqft.
    - 'LotFrontage' above 300 sqft."""

    outliers = df[(df['GrLivArea']>4000)|(df['SalePrice']>700000)|(df['LotArea']>100000)|(df['LotFrontage']>300)].index
    df.drop(index=outliers, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#----------------------------------------------------------------------------------

def drop_features_tr(df):
    """This function drops the features that are missing more than half of the values."""
    missing_values_dict={}
    for column in df.columns:
        percentage_missing = (df[column].isna().sum()/len(df[column]))*100
        if percentage_missing > 50:
            missing_values_dict[column] = percentage_missing
            features = pd.Series(missing_values_dict).sort_values(ascending=True)
            features_index = features.index

    df.drop(columns=features_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, features_index

#----------------------------------------------------------------------------------

def drop_features_ts(df, drop_ft_obj):
    """This function drops the features."""
    df.drop(columns=drop_ft_obj, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#----------------------------------------------------------------------------------

def data_imputation_tr(df):
    """Handles data imputation for train data, managing missing values within the dataset.
    Also, returns dictionaries containing median values for:
    - Median masonary veneer area
    - Median lot frontage area"""

    df['BsmtQual'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtQual'].isna())), 'N/A', df['BsmtQual'])
    df['BsmtCond'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtCond'].isna())), 'N/A', df['BsmtCond'])
    df['BsmtExposure'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtExposure'].isna())), 'N/A', df['BsmtExposure'])
    df['BsmtFinType1'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtFinType1'].isna())), 'N/A', df['BsmtFinType1'])
    df['BsmtFinType2'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtFinType2'].isna())), 'N/A', df['BsmtFinType2'])
    df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)

    df['GarageType'] = np.where(((df['GarageArea']==0)&(df['GarageType'].isna())), 'N/A', df['GarageType'])
    df['GarageYrBlt'] = np.where(((df['GarageArea']==0)&(df['GarageYrBlt'].isna())), 0, df['GarageYrBlt'])
    df['GarageFinish'] = np.where(((df['GarageArea']==0)&(df['GarageFinish'].isna())), 'N/A', df['GarageFinish'])
    df['GarageQual'] = np.where(((df['GarageArea']==0)&(df['GarageQual'].isna())), 'N/A', df['GarageQual'])
    df['GarageCond'] = np.where(((df['GarageArea']==0)&(df['GarageCond'].isna())), 'N/A', df['GarageCond'])

    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    df['FireplaceQu'].fillna('N/A', inplace=True)

    median_masonary = df.groupby('Neighborhood')['MasVnrArea'].median()
    median_msonary_dictionary = median_masonary.to_dict()

    median_lotfront = df.groupby('Neighborhood')['LotFrontage'].median()
    median_lotfront_dictionary = median_lotfront.to_dict()

    df['MasVnrArea'] = df.apply(lambda row: median_msonary_dictionary.get(row['Neighborhood'], row['MasVnrArea']) if pd.isna(row['MasVnrArea']) else row['MasVnrArea'],axis=1)
    df['LotFrontage'] = df.apply(lambda row: median_lotfront_dictionary.get(row['Neighborhood'], row['LotFrontage']) if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1)
    return df, median_msonary_dictionary, median_lotfront_dictionary

#----------------------------------------------------------------------------------

def data_imputation_ts(df, mm_obj, lf_obj):
    """Handles data imputation for test data, managing missing values within the dataset."""

    df['BsmtQual'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtQual'].isna())), 'N/A', df['BsmtQual'])
    df['BsmtCond'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtCond'].isna())), 'N/A', df['BsmtCond'])
    df['BsmtExposure'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtExposure'].isna())), 'N/A', df['BsmtExposure'])
    df['BsmtFinType1'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtFinType1'].isna())), 'N/A', df['BsmtFinType1'])
    df['BsmtFinType2'] = np.where(((df['TotalBsmtSF']==0)&(df['BsmtFinType2'].isna())), 'N/A', df['BsmtFinType2'])
    df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)

    df['GarageType'] = np.where(((df['GarageArea']==0)&(df['GarageType'].isna())), 'N/A', df['GarageType'])
    df['GarageYrBlt'] = np.where(((df['GarageArea']==0)&(df['GarageYrBlt'].isna())), 0, df['GarageYrBlt'])
    df['GarageFinish'] = np.where(((df['GarageArea']==0)&(df['GarageFinish'].isna())), 'N/A', df['GarageFinish'])
    df['GarageQual'] = np.where(((df['GarageArea']==0)&(df['GarageQual'].isna())), 'N/A', df['GarageQual'])
    df['GarageCond'] = np.where(((df['GarageArea']==0)&(df['GarageCond'].isna())), 'N/A', df['GarageCond'])

    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
    df['FireplaceQu'].fillna('N/A', inplace=True)

    median_msonary_dictionary = mm_obj
    median_lotfront_dictionary = lf_obj

    df['MasVnrArea'] = df.apply(lambda row: median_msonary_dictionary.get(row['Neighborhood'], row['MasVnrArea']) if pd.isna(row['MasVnrArea']) else row['MasVnrArea'],axis=1)
    df['LotFrontage'] = df.apply(lambda row: median_lotfront_dictionary.get(row['Neighborhood'], row['LotFrontage']) if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1)
    return df

#----------------------------------------------------------------------------------

def data_org(df):
    """Organizes the dataset by performing the following steps:
    - Converts binary features with 'yes' or 'no' values into binary numeric format.
    - Replaces categorical ordinal values with corresponding numeric representations.
    - Creates a 'quality condition' feature by averaging quality and condition attributes.
    - Aggregates basement finish types into a single feature for enhanced clarity."""

    df['MSSubClass'] = df['MSSubClass'].astype('object')

    df['GarageFinish'] = df['GarageFinish'].map(lambda x: 1 if x=='Fin' else 0)
    df['CentralAir'] = df['CentralAir'].map(lambda x: 1 if x=='Y' else 0)
    df['Functional'] = df['Functional'].map(lambda x: 1 if x=='Typ' else 0)
    df['PavedDrive'] = df['PavedDrive'].map(lambda x: 1 if x=='Y' else 0)
    df['Fireplaces'] = df['Fireplaces'].map(lambda x: 1 if x>0 else 0)
    df['Street'] = df['Street'].map(lambda x: 1 if x=='Pave' else 0)
    df['Utilities'] = df['Utilities'].map(lambda x: 1 if x=='AllPub' else 0)

    qc_dictionary = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'N/A':0}
    height_dictionary = {'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60, 'N/A':0}

    df['ExterQual'] = df['ExterQual'].replace(qc_dictionary)
    df['ExterCond'] = df['ExterCond'].replace(qc_dictionary)
    df['BsmtCond'] = df['BsmtCond'].replace(qc_dictionary)
    df['HeatingQC'] = df['HeatingQC'].replace(qc_dictionary)
    df['KitchenQual'] = df['KitchenQual'].replace(qc_dictionary)
    df['FireplaceQu'] = df['FireplaceQu'].replace(qc_dictionary)
    df['GarageQual'] = df['GarageQual'].replace(qc_dictionary)
    df['GarageCond'] = df['GarageCond'].replace(qc_dictionary)

    df['BsmtHeight'] = df['BsmtQual'].replace(height_dictionary)
    df.drop(columns='BsmtQual', inplace=True)

    df['OverallQC'] = (df['OverallQual'] + df['OverallCond'])/2
    df['ExteriorQC'] = (df['ExterQual'] + df['ExterCond'])/2
    df['GarageQC'] = (df['GarageQual'] + df['GarageCond'])/2

    df['BsmtFinType1'] = df['BsmtFinType1'].map(lambda x: 1 if x=='GLQ' else 1 if x=='ALQ' else 0)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(lambda x: 1 if x=='GLQ' else 1 if x=='ALQ' else 0)

    df['BsmtFinish'] = df['BsmtFinType1'] + df['BsmtFinType2']

    df.drop(columns=['OverallQual','OverallCond','ExterQual','ExterCond','GarageQual','GarageCond','BsmtExposure','BsmtFinType1','BsmtFinType2'], inplace=True)
    return df

#----------------------------------------------------------------------------------

def test_fill_values(df):
    df['Exterior1st'].fillna('VinylSd', inplace=True)
    df['Exterior2nd'].fillna('VinylSd', inplace=True)
    df['BsmtCond'].fillna(3, inplace=True)
    df['BsmtFinSF1'].fillna(379.5, inplace=True)
    df['BsmtFinSF2'].fillna(0, inplace=True)
    df['BsmtUnfSF'].fillna(482.5, inplace=True)
    df['TotalBsmtSF'].fillna(992, inplace=True)
    df['BsmtFullBath'].fillna(0, inplace=True)
    df['BsmtHalfBath'].fillna(0, inplace=True)
    df['KitchenQual'].fillna(3, inplace=True)
    df['GarageYrBlt'].fillna(1979, inplace=True)
    df['GarageCars'].fillna(2, inplace=True)
    df['GarageArea'].fillna(477.5, inplace=True)
    df['SaleType'].fillna('WD', inplace=True)
    df['BsmtHeight'].fillna(80, inplace=True)
    df['GarageQC'].fillna(6, inplace=True)
    return df

#----------------------------------------------------------------------------------

def feature_eng_tr(df):
    """Generates new features within the train dataset and provides a dictionary with average neighborhood decibel levels. The function returns the updated dataset along with a dictionary containing average decibel levels for each neighborhood."""
    # Total Porch Area
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

    # Total Living Area
    df['TotalLivSF'] = df['MasVnrArea'] + df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea'] + df['PoolArea'] + df['TotalPorchSF'] + df['WoodDeckSF']

    # New Built House
    df['NewHouse'] = df['SaleType'].map(lambda x: 1 if x=='New' else 0)

    # Expensive Neighborhood
    df['ExpNeighborhood'] = df['Neighborhood'].map(lambda x: 1 if x=='NoRidge' else 1 if x=='NridgHt' else 1 if x=='StoneBr' else 0)

    # Basement and Ground Bathrooms
    df['BsmtHalfBath'] = df['BsmtHalfBath'].map(lambda x: 0.5 if x==1 else 1 if x==2 else 0)
    df['HalfBath'] = df['HalfBath'].map(lambda x: 0.5 if x==1 else 1 if x==2 else 0)

    df['BsmtBaths'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df['GrBaths'] = df['FullBath'] + df['HalfBath']

    # Age of the House
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    #Neighborhood Noise dB
    condition = {'Norm':'Normal', 'Feedr':'Road', 'PosN':'Good', 'Artery':'Road', 'RRAe':'Railroad', 'RRNn':'Railroad', 'RRAn':'Railroad',
                'PosA':'Good', 'RRNe':'Railroad'}
    df['Condition1'] = df['Condition1'].replace(condition)
    df['Condition2'] = df['Condition2'].replace(condition)
    df['NeighborCondition'] = df['Condition1'] + df['Condition2']

    condition2 = {'NormalNormal':60, 'RoadNormal':75, 'GoodNormal':55, 'RoadRoad':80,'RailroadNormal':85,
    'RoadRailroad':100, 'RailroadRoad':100, 'GoodGood':50,'RoadGood':70}
    df['NeighborNoise(dB)'] = df['NeighborCondition'].replace(condition2)

    avgdB = df.groupby('Neighborhood')['NeighborNoise(dB)'].mean()
    avg_dB_dict = avgdB.to_dict()

    df['NeighborAvg_dB'] = df['Neighborhood'].replace(avg_dB_dict)
    df['NeighborAvg_dB'] = round(df['NeighborAvg_dB'],2)

    df.drop(columns=['MoSold','SaleType','SaleCondition','Condition1','Condition2','NeighborNoise(dB)','NeighborCondition','BsmtFullBath','BsmtHalfBath','YrSold','LotConfig','HalfBath'], inplace=True)
    return df, avg_dB_dict

#----------------------------------------------------------------------------------

def feature_eng_ts(df, dB_obj):
    """Generates new features within the test dataset and provides a dictionary with average neighborhood decibel levels. The function returns the updated dataset along with a dictionary containing average decibel levels for each neighborhood."""

    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['TotalLivSF'] = df['MasVnrArea'] + df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea'] + df['PoolArea'] + df['TotalPorchSF'] + df['WoodDeckSF']
    df['NewHouse'] = df['SaleType'].map(lambda x: 1 if x=='New' else 0)
    df['ExpNeighborhood'] = df['Neighborhood'].map(lambda x: 1 if x=='NoRidge' else 1 if x=='NridgHt' else 1 if x=='StoneBr' else 0)
    df['BsmtHalfBath'] = df['BsmtHalfBath'].map(lambda x: 0.5 if x==1 else 1 if x==2 else 0)
    df['HalfBath'] = df['HalfBath'].map(lambda x: 0.5 if x==1 else 1 if x==2 else 0)
    df['BsmtBaths'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df['GrBaths'] = df['FullBath'] + df['HalfBath']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    condition = {'Norm':'Normal', 'Feedr':'Road', 'PosN':'Good', 'Artery':'Road', 'RRAe':'Railroad', 'RRNn':'Railroad', 'RRAn':'Railroad',
                'PosA':'Good', 'RRNe':'Railroad'}
    df['Condition1'] = df['Condition1'].replace(condition)
    df['Condition2'] = df['Condition2'].replace(condition)
    df['NeighborCondition'] = df['Condition1'] + df['Condition2']

    condition2 = {'NormalNormal':60, 'RoadNormal':75, 'GoodNormal':55, 'RoadRoad':80,'RailroadNormal':85,
    'RoadRailroad':100, 'RailroadRoad':100, 'GoodGood':50,'RoadGood':70}
    df['NeighborNoise(dB)'] = df['NeighborCondition'].replace(condition2)

    avg_dB_dict = dB_obj

    df['NeighborAvg_dB'] = df['Neighborhood'].replace(avg_dB_dict)
    df['NeighborAvg_dB'] = round(df['NeighborAvg_dB'],2)

    df.drop(columns=['MoSold','SaleType','SaleCondition','Condition1','Condition2','NeighborNoise(dB)','NeighborCondition','BsmtFullBath','BsmtHalfBath','YrSold','LotConfig','HalfBath'], inplace=True)
    return df

#----------------------------------------------------------------------------------

def onehot_encode_tr(df):
    """Performs One Hot Encoding on the train dataset and provides the encoded dataframe along with the encoder."""
    categorical_ft = df[df.select_dtypes('object').columns]
    oneHot = ce.OneHotEncoder(use_cat_names=True)
    oneHot.fit(categorical_ft)
    encoded_train = oneHot.transform(categorical_ft)
    return encoded_train, oneHot

#----------------------------------------------------------------------------------

def onehot_encode_ts(df, onehot_obj):
    """Performs One Hot Encoding on the test dataset."""
    categorical_ft = df[df.select_dtypes('object').columns]
    encoded_test = onehot_obj.transform(categorical_ft)
    return encoded_test

#----------------------------------------------------------------------------------

def selected_features_tr(df, encoded_df):
    """Returns selected features from the train dataset where the correlation with the target is greater than 0.1."""
    target_ft = df[['SalePrice']]
    encoded_train_sub = encoded_df.join(target_ft)
    encoded_corr = encoded_train_sub.corr()[['SalePrice']].reset_index().rename(columns={'index':'Feature', 'SalePrice':'Correlation'})
    selected_corr = encoded_corr[abs(encoded_corr['Correlation'])>0.1]
    selected_ft = np.delete(selected_corr['Feature'].values, [-1])
    selected_encoded_df = encoded_df[selected_ft]
    return selected_encoded_df, selected_ft

#----------------------------------------------------------------------------------

def selected_features_ts(encoded_df, ft_obj):
    """Returns selected features from the test dataset."""
    selected_encoded_df = encoded_df[ft_obj]
    return selected_encoded_df

#----------------------------------------------------------------------------------

def scale_data_tr(df):
    df.drop(columns=['Street', 'Utilities'], inplace=True)
    non_scaled = df[['YearBuilt', 'YearRemodAdd', 'BsmtCond', 'HeatingQC', 'CentralAir', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional',
                    'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'PavedDrive', 'OverallQC', 'ExteriorQC', 'GarageQC',
                    'BsmtFinish', 'NewHouse', 'ExpNeighborhood', 'BsmtBaths', 'GrBaths']]
    need_scale = df[df.select_dtypes(['float', 'int']).columns].drop(columns=non_scaled.columns)
    need_scale.drop(columns='SalePrice', inplace=True)
    std_SC = StandardScaler()
    scaled_df = pd.DataFrame(std_SC.fit_transform(need_scale), columns=need_scale.columns)
    return scaled_df, non_scaled, std_SC

#----------------------------------------------------------------------------------

def scale_data_ts(df, scale_obj):
    """Scales the test dataset."""
    try:
        df.drop(columns=['Street', 'Utilities','SalePrice'], inplace=True)
        non_scaled = df[['YearBuilt', 'YearRemodAdd', 'BsmtCond', 'HeatingQC', 'CentralAir', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional',
                            'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'PavedDrive', 'OverallQC', 'ExteriorQC', 'GarageQC',
                            'BsmtFinish', 'NewHouse', 'ExpNeighborhood', 'BsmtBaths', 'GrBaths']]

        need_scale = df[df.select_dtypes(['float', 'int']).columns].drop(columns=non_scaled.columns)

        scaled_df = pd.DataFrame(scale_obj.transform(need_scale), columns=need_scale.columns)
    except:
        df.drop(columns=['Street', 'Utilities'], inplace=True)
        non_scaled = df[['YearBuilt', 'YearRemodAdd', 'BsmtCond', 'HeatingQC', 'CentralAir', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional',
                            'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'PavedDrive', 'OverallQC', 'ExteriorQC', 'GarageQC',
                            'BsmtFinish', 'NewHouse', 'ExpNeighborhood', 'BsmtBaths', 'GrBaths']]

        need_scale = df[df.select_dtypes(['float', 'int']).columns].drop(columns=non_scaled.columns)

        scaled_df = pd.DataFrame(scale_obj.transform(need_scale), columns=need_scale.columns)
    return scaled_df, non_scaled

#----------------------------------------------------------------------------------

def final_train(df, encoded_df, scaled_df, non_scaled):
    """Provides the final train dataset prepared for model training."""
    target = df[['SalePrice']]
    train_final = encoded_df.join(scaled_df)
    train_final = train_final.join(non_scaled)
    train_final = train_final.join(target)
    return train_final

#----------------------------------------------------------------------------------

def final_test(encoded_df, scaled_df, non_scaled):
    """Provides the final test dataset."""
    test_final = encoded_df.join(scaled_df)
    test_final = test_final.join(non_scaled)
    return test_final

#----------------------------------------------------------------------------------

def transform_target(df):
    """Applies a logarithmic transformation to the sale price in the dataset."""
    df['SalePrice_trans'] = np.log(df['SalePrice'])
    return df

#----------------------------------------------------------------------------------

def train_test_df(df):
    """Returns the x_train, x_test, y_train, and y_test dataframes after splitting the dataset into training and testing sets."""
    x = df.drop(columns=['SalePrice', 'SalePrice_trans'])
    y = df['SalePrice_trans']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

#----------------------------------------------------------------------------------

def validation_data_split(df):
    """Splits the validation dataset into dependent and independent variables."""
    x = df.drop(columns=['SalePrice', 'SalePrice_trans'])
    y = df['SalePrice_trans']
    return x, y

#----------------------------------------------------------------------------------

def train_models(x_train, y_train):
    """Trains the following models:
    - ElasticNet
    - GradientBoostingRegressor
    - XGBRegressor"""
    en = ElasticNet(alpha = 0.001, l1_ratio=0.99)
    gbr = GradientBoostingRegressor(n_estimators=500)
    xgbr = xgb.XGBRegressor(n_estimators=2900, max_depth=2, learning_rate=0.01)

    en.fit(x_train, y_train)
    gbr.fit(x_train, y_train)
    xgbr.fit(x_train, y_train)

    return en, gbr, xgbr

#----------------------------------------------------------------------------------

def price_prediction(x_test, en, gbr, xgbr):
    """Predicts house prices and returns predictions following a specified order.
    - ElasticNet
    - GradientBoostingRegressor
    - XGBRegressor
    """
    en_pred = en.predict(x_test)
    gbr_pred = gbr.predict(x_test)
    xgbr_pred = xgbr.predict(x_test)
    return en_pred, gbr_pred, xgbr_pred

#----------------------------------------------------------------------------------

def rmse_result(y_test, en_pred, gbr_pred, xgbr_pred):
    """Returns the Root Mean Squared Error (RMSE) dataframe of the predictions compared to the actual values, following a specific order."""
    result = pd.DataFrame({
    "ElasticNet": {'RMSE':mean_squared_error(np.exp(y_test), np.exp(en_pred)) ** .5},
    'GradientBoost': {'RMSE':mean_squared_error(np.exp(y_test), np.exp(gbr_pred)) ** .5},
    "XGBoost": {'RMSE':mean_squared_error(np.exp(y_test), np.exp(xgbr_pred)) ** .5},
    })
    return result

#----------------------------------------------------------------------------------

def weighted_pred_tr(y_test, en_pred, gbr_pred, xgbr_pred):
    """Returns the weighted prediction calculated from the train dataset."""
    weighted_pred = en_pred * 0.5 + gbr_pred * 0.2 + xgbr_pred * 0.3
    y_test_inv = np.exp(y_test)
    return weighted_pred, y_test_inv

#----------------------------------------------------------------------------------

def weighted_pred_ts(en_pred, gbr_pred, xgbr_pred):
    """Returns the weighted prediction calculated from the test dataset."""
    weighted_pred = en_pred * 0.5 + gbr_pred * 0.2 + xgbr_pred * 0.3
    result = pd.DataFrame(np.exp(weighted_pred), columns=['SalePrice'])
    return result

#----------------------------------------------------------------------------------

def print_weighte_pred_rmse(weighted_pred, y_test_inv):
    """Prints the Root Mean Squared Error (RMSE) of the weighted prediction calculated from the provided dataset."""
    print('Weighted RMSE:',mean_squared_error(y_test_inv, np.exp(weighted_pred)) ** .5)

#----------------------------------------------------------------------------------

def pickle_objects(obj, str_name):
    """Saves the object using pickle serialization with the specified filename."""
    pickle.dump(obj, open(str_name + ".pickle", "wb"))
    print("Pickle save complete!")

#----------------------------------------------------------------------------------

def load_objects(path):
    """Loads the object previously saved using pickle serialization from the specified filename."""
    obj = pickle.load(open(path, "rb"))
    return obj