from y_set_df_dm_dd import dm_df_dd_list
from x_set_creator import sensor_stdev,feature_vector,sensor_mean,sensor_median_high
import pandas as pd
import numpy as np

from y_set_df_dm_dd import dm_df_dd_list

data = pd.DataFrame({'defects':dm_df_dd_list})
data = data.assign(**sensor_median_high)

#data = pd.DataFrame({'defects':dm_df_dd_list,'sensor':feature_vector})

for i in range (0, len(data)):
    if data['defects'][i] != 'df' and data['defects'][i] != 'dm' and data['defects'][i] != 'dd':
        data = data.drop([i])


