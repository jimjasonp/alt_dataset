from y_set_df_dm_dd import dm_df_dd_list
from x_set_creator import sensor_stdev,feature_vector,sensor_mean,sensor_median_high,sensor_max
import pandas as pd
import numpy as np

from y_set_df_dm_dd import dm_df_dd_list

data = pd.DataFrame({
    'defects':dm_df_dd_list,

    's2_mean':sensor_mean['s2'],
    's3_mean':sensor_mean['s3'],
    's4_mean':sensor_mean['s4'],

    's2_max':sensor_max['s2'],
    's3_max':sensor_max['s3'],
    's4_max':sensor_max['s4'],

    's2_median_high':sensor_median_high['s2'],
    's3_median_high':sensor_median_high['s3'],
    's4_median_high':sensor_median_high['s4'],

    's2_stdev':sensor_stdev['s2'],
    's3_stdev':sensor_stdev['s3'],
    's4_stdev':sensor_stdev['s4'],
    })
#data = pd.DataFrame({'defects':dm_df_dd_list,'sensor':feature_vector})

for i in range (0, len(data)):
    if data['defects'][i] != 'df' and data['defects'][i] != 'dm' and data['defects'][i] != 'dd':
        data = data.drop([i])


