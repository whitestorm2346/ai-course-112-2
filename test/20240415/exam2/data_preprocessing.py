import pandas as pd

class DataPreprocessing:
    def __init__(self) -> None:
        pass

    def csv_data_mapping(self, data: pd.DataFrame): # pd.DataFrame, dict{dict}
        return data

    def csv_data_cleaning(self, data: pd.DataFrame, 
                          missing_data_cleaning_method='delete', 
                          abnormal_data_cleaning_method='delete') -> pd.DataFrame:
        if missing_data_cleaning_method == 'delete':
            data = data.dropna()
        elif missing_data_cleaning_method == 'zero':
            data = data.fillna(0)
        elif missing_data_cleaning_method == 'median':
            pass
        elif missing_data_cleaning_method == 'plural':
            pass
        else:
            print('missing_data_cleaning_method undefined value')
            exit(1)

        if abnormal_data_cleaning_method == 'delete':
            pass
        elif abnormal_data_cleaning_method == 'zero':
            pass
        elif abnormal_data_cleaning_method == 'median':
            pass
        elif abnormal_data_cleaning_method == 'plural':
            pass
        else:
            print('missing_data_cleaning_method undefined value')
            exit(1)

        return data