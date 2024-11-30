from y_set_df_dm_dd import dm_df_dd_list
import pandas as pd
import glob
import os
import numpy as np
import statistics
path = r'C:\Users\jimja\Desktop\thesis\Balanced_Data'
#path = r'C:\Users\jimja\Desktop\thesis\data' # use your path
# to sensor data list einai auto pou einai sth morfh gia train
sensor_data_list = []
name_list = []


### h lista tou damage percnetage kai twn dedomenwn apo sensores den exoun idio megethos giati leipoun kapoia data apo damage
### gia auto to logo oi arithmoi instance pou den uparxoun kai stis duo listes tha fugoun

def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    return power_spectrum


# gia kathe filename sto path pou tou exw dwsei afairei to .csv wste meta na mporei na diabasei ton arithmo
for filename in sorted(glob.glob(os.path.join(path , "data*"))):
    filename = filename.removesuffix('.csv')
    name_list.append(filename)


#apo kathe filename krataei mono ton arithmo sto telos kai me auton ton arithmo ftiaxeni th nea sthlh index number
sensor_data = pd.DataFrame({'name':name_list})
sensor_data['sensor_index_number'] = [int(i.split('_')[-1]) for i in sensor_data['name']]


#kanw sort th lista basei tou index number
sensor_data = sensor_data.sort_values(by=['sensor_index_number'])


suffix='.csv'
new_names=[]


#se kathe filename sth lista pou exei ginei sort prosthetei to .csv wste na mporei na to diabasei
for filename in sensor_data['name']:
    filename = filename+suffix
    new_names.append(filename)



#anoigei ta arxeia apo kathe path kai ftiaxnei th lista me tis metrhseis

for filename in new_names:
    df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
    sensor_data_list.append(df)

sensor_max = pd.DataFrame()



sensor_fft_df = pd.DataFrame()
sensor_names = ['s2','s3','s4']
for sensor in sensor_names:
    sensor_fft = []
    #gia kathe sample sensora dld gia kathe xronoseira (pou prokuptei apo to shma pou lambanei o sensoras efarmozo fft
    for i in range(0,len(sensor_data_list)):
        #efarmozo to metasxhmatismo fourier (fft) se kathe timeserie
        sample_sensor =sensor_data_list[i][sensor]
        #sample_sensor = np.array(sample_sensor,dtype=float)
        sample_sensor = fourier(sample_sensor)
        sensor_fft.append(sample_sensor)
    # tis times tou kathe feature tis pernaw se ena df 
    new_data = {sensor: sensor_fft}
    sensor_fft_df = sensor_fft_df.assign(**new_data)
#kataskeuazw ena dataframe gia to kathe feature me to antistoixo onoma
sensor_max = sensor_max.assign(**sensor_fft_df)


data = pd.DataFrame({'defects':dm_df_dd_list})

data = data.assign(**sensor_max)


for i in range (0, len(data)):
    if data['defects'][i] != 'df' and data['defects'][i] != 'dm' and data['defects'][i] != 'dd':
        data = data.drop([i])

print(data['defects'])