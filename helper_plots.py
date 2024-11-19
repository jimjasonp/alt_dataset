import matplotlib.pyplot as plt
from x_set_creator import sensor_data_list,power_spectrum_list
from y_set_df_dm_dd import dm_df_dd_list

sensor_samples = [sensor_data_list[100]['s2'], #dd
         sensor_data_list[3]['s2'], #df dm
         sensor_data_list[4]['s2'], # df
         sensor_data_list[0]['s2'], #clean
         sensor_data_list[149]['s2'],# df dm dd
         sensor_data_list[65]['s2']] #dm

power_samples = [power_spectrum_list[100], ##dd
                 power_spectrum_list[3], # df dm
                 power_spectrum_list[4], # df
                 power_spectrum_list[0], #clean
                 power_spectrum_list[149], # df dm dd
                 power_spectrum_list[65]] # dm

### Raw signal


plt.subplot(2,3,1)
plt.plot(sensor_samples[0])
plt.title("Raw signal with delamination")
plt.grid(True)


plt.subplot(2,3,2)
plt.plot(sensor_samples[1])
plt.title("Raw signal with matrix and fiber defect")
plt.grid(True)


plt.subplot(2,3,3)
plt.plot(sensor_samples[2])
plt.title("Raw signal with fiber defect")
plt.grid(True)


plt.subplot(2,3,4)
plt.plot(sensor_samples[3])
plt.title("Raw signal without defects")
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(sensor_samples[4])
plt.title("Raw signal with delamination matrix and fiber defects")
plt.grid(True)


plt.subplot(2,3,6)
plt.plot(sensor_samples[5])
plt.title("Raw signal with matrix defect")
plt.grid(True)
plt.show()


# FFT


plt.subplot(2,3,1)
plt.plot(power_samples[0])
plt.title("FFT signal with delamination")
plt.xlim(0,750)
plt.yscale("log")
plt.grid(True)


plt.subplot(2,3,2)
plt.plot(power_samples[1])
plt.title("FFT signal with matrix and fiber defect")
plt.xlim(0,750)
plt.yscale("log")
plt.grid(True)


plt.subplot(2,3,3)
plt.plot(power_samples[2])
plt.title("FFT signal with fiber defect")
plt.yscale("log")
plt.xlim(0,750)
plt.grid(True)


plt.subplot(2,3,4)
plt.plot(power_samples[3])
plt.title("FFT signal without defects")
plt.yscale("log")
plt.xlim(0,750)
plt.grid(True)

plt.subplot(2,3,5)
plt.plot(power_samples[4])
plt.title("FFT signal with delamination matrix and fiber defects")
plt.yscale("log")
plt.xlim(0,750)
plt.grid(True)


plt.subplot(2,3,6)
plt.plot(power_samples[5])
plt.title("FFT signal with matrix defect")
plt.yscale("log")
plt.xlim(0,750)
plt.grid(True)
plt.show()