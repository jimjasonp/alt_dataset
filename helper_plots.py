import matplotlib.pyplot as plt
from x_set_creator import sensor_data_list,power_spectrum_list
from y_set_df_dm_dd import dm_df_dd_list

sensor_samples = [sensor_data_list[100]['s2'], #dd
         sensor_data_list[3]['s2'], #df dm
         sensor_data_list[4]['s2'], # df
         sensor_data_list[0]['s2'], #clean
         sensor_data_list[149]['s2'],# df dm dd
         sensor_data_list[65]['s2'],

         sensor_data_list[100]['s3'], #dd
         sensor_data_list[3]['s3'], #df dm
         sensor_data_list[4]['s3'], # df
         sensor_data_list[0]['s3'], #clean
         sensor_data_list[149]['s3'],# df dm dd
         sensor_data_list[65]['s3'],
        
         sensor_data_list[100]['s4'], #dd
         sensor_data_list[3]['s4'], #df dm
         sensor_data_list[4]['s4'], # df
         sensor_data_list[0]['s4'], #clean
         sensor_data_list[149]['s4'],# df dm dd
         sensor_data_list[65]['s4'] ]#dm

power_samples = [power_spectrum_list[251], ##dd
                 power_spectrum_list[254], # df dm
                 power_spectrum_list[255], # df
                 power_spectrum_list[151], #clean
                 power_spectrum_list[300], # df dm dd
                 power_spectrum_list[216],

                 power_spectrum_list[100], ##dd
                 power_spectrum_list[103], # df dm
                 power_spectrum_list[104], # df
                 power_spectrum_list[0], #clean
                 power_spectrum_list[149], # df dm dd
                 power_spectrum_list[65],

                 power_spectrum_list[402], ##dd
                 power_spectrum_list[405], # df dm
                 power_spectrum_list[406], # df
                 power_spectrum_list[302], #clean
                 power_spectrum_list[451], # df dm dd
                 power_spectrum_list[367]] # dm

raw_titles = ["Raw signal with delamination",
          "Raw signal with matrix and fiber defect",
          "Raw signal with fiber defect",
          "Raw signal without defects",
          "Raw signal with delamination matrix and fiber defects",
          "Raw signal with matrix defect"]


fft_titles = ["FFT signal with delamination",
          "FFT signal with matrix and fiber defect",
          "FFT signal with fiber defect",
          "FFT signal without defects",
          "FFT signal with delamination matrix and fiber defects",
          "FFT signal with matrix defect"]


### Raw signal

for i in range(0,len(fft_titles)):
    plt.subplot(2,3,i+1)
    plt.plot(sensor_samples[i])
    plt.plot(sensor_samples[i+6])
    plt.plot(sensor_samples[i+12])
    plt.title(raw_titles[i])
    plt.grid(True)
plt.show()


for i in range(0,len(fft_titles)):
    plt.subplot(2,3,i+1)
    plt.plot(power_samples[i])
    plt.plot(power_samples[i+6])
    plt.plot(power_samples[i+12])
    plt.yscale('log')
    plt.xlim(0,350)
    plt.title(fft_titles[i])
    plt.grid(True)
plt.show()
