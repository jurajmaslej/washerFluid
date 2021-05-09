data_folder = 'data'
source_file = 'TrainingDataset_BI.xlsx'
sheet_name = 'DATA'
csv_name = 'train.csv'
feature_analysis = 'graphs/feature_analysis'
correlation_analysis = 'graphs/correlation'
models_regression = 'graphs/regression'
models_svm = 'graphs/svm'

features_all = ['SegmentId', 'TankAge', 'IsEvent',
       'IsCombinedHoliday', 'IsCombinedVacations', 'DaysBA_MigrHoliday',
       'DaysBA_MigrVacation', 'DaysA_WaterChange', 'Target', 'VolumeAvg1D',
       'VolumeAvg3D', 'VolumeAvg7D', 'IsDry', 'IsSleet', 'IsRain', 'IsSnow',
       'PrecipIntCoef', 'PrecipIntCoefAVG1D', 'PrecipIntCoefAVG3D',
       'PrecipIntCoefAVG7D', 'PrecipIntCoefAVG14D', 'PrecipIntCoefAVG30D',
       'PrecipCount1D', 'PrecipCount3D', 'PrecipCount7D', 'PrecipCount14D',
       'PrecipCount30D', 'TemperatureMin', 'TemperatureMax', 'IsBelowDP',
       'BelowDPCount1D', 'BelowDPCount3D', 'BelowDPCount7D', 'BelowDPCount14D',
       'BelowDPCount30D', 'Humidity', 'Visibility', 'WindSpeed', 'CloudCover',
       'Pressure']

features_numeric = ['SegmentId', 'TankAge', 'DaysBA_MigrHoliday',
       'DaysBA_MigrVacation', 'DaysA_WaterChange', 'Target', 'VolumeAvg1D',
       'VolumeAvg3D', 'VolumeAvg7D', 'PrecipIntCoef', 'PrecipIntCoefAVG1D', 'PrecipIntCoefAVG3D',
       'PrecipIntCoefAVG7D', 'PrecipIntCoefAVG14D', 'PrecipIntCoefAVG30D',
       'PrecipCount1D', 'PrecipCount3D', 'PrecipCount7D', 'PrecipCount14D',
       'PrecipCount30D', 'TemperatureMin', 'TemperatureMax', 'BelowDPCount1D',
       'BelowDPCount3D', 'BelowDPCount7D', 'BelowDPCount14D',
       'BelowDPCount30D', 'Humidity', 'Visibility', 'WindSpeed', 'CloudCover',
       'Pressure']

features_categorical = ['IsEvent', 'IsCombinedHoliday', 'IsCombinedVacations', 'IsDry',
                        'IsSleet', 'IsRain', 'IsSnow', 'IsBelowDP']

features_target = ['Target',  'VolumeAvg1D', 'VolumeAvg3D', 'VolumeAvg7D']


features_high_correlation = ['TemperatureMin', 'TemperatureMax', 'Humidity', 'IsRain', 'IsSnow',
                              'IsCombinedVacations', 'DaysBA_MigrVacation']  # 'WindSpeed',

features_high_correlation_with_volume = ['VolumeAvg3D', 'VolumeAvg7D',
                                         'TemperatureMin', 'TemperatureMax',
                                          'IsRain', 'IsSnow', 'DaysBA_MigrVacation']  # 'WindSpeed',

features_speed_run = [ 'VolumeAvg7D','VolumeAvg3D','TemperatureMin',  'DaysBA_MigrVacation']

features_neural_net = [ 'VolumeAvg7D','VolumeAvg3D', 'VolumeAvg1D', 'DaysBA_MigrHoliday',
                       'DaysBA_MigrVacation', 'DaysA_WaterChange','TemperatureMin', 'TemperatureMax',
                       'IsRain', 'IsSnow']

features_highest_pca = ['SegmentId','TankAge','DaysBA_MigrHoliday', 'DaysBA_MigrVacation',
                        'DaysA_WaterChange','VolumeAvg1D','VolumeAvg3D',
                        'PrecipIntCoef', 'PrecipIntCoefAVG1D', 'PrecipIntCoefAVG3D',
                        'TemperatureMin', 'TemperatureMax']

features_highest_pca_mini = ['SegmentId','TankAge','DaysBA_MigrHoliday', 'DaysBA_MigrVacation',
                        'DaysA_WaterChange','VolumeAvg1D','VolumeAvg3D',
                        'PrecipIntCoef']