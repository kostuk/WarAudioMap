import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Generate  function to return  jump_x and jump_y
# input audio and sample_rate get code from about

def get_auduo_jump_power(audio, sample_rate, show_chart=False):
    # Calcaluate mel spectrogram in db
    mel_stft_audio = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    spectrogram = librosa.amplitude_to_db(mel_stft_audio)

    # energy of  spectrogram
    energy = np.sum(spectrogram, axis=0)/spectrogram.shape[0]
    # Speed of spectrogram change and derivative(convolve in 5 winodw) and append zeros at the beginning and end
    window_size = 5
    deriv = np.convolve(np.diff(energy), np.ones(window_size)/window_size, mode='valid')
    deriv = np.insert(  deriv,  0, np.zeros(window_size))

#    deriv = np.insert(  deriv,  0, np.zeros(window_size//2))
#    deriv = np.append(  deriv,  np.zeros(window_size- window_size//2))

    # Set a threshold to detect spikes (2 times higher than the STD from  average value)
    threshold_mean =  np.mean(energy)
    threshold = threshold_mean +1.5*np.std(energy)
    threshold = np.max(energy)-(np.max(energy)-np.min(energy))*0.4

    threshold_speed_mean =  np.mean(deriv)
    threshold_speed = threshold_speed_mean +1.5*np.std(deriv)

    # Convert energy to time
    jump_times = librosa.times_like(energy, sr=sample_rate)

    # get index of spikes
    jump_indices = np.where(energy > threshold)[0]
    
    # get time of spikes of maximal energy and maxmum speed
    jump_x = []
    jump_y = []
    speed_direction = 0
    is_front = 0
    speed_limit = []
    value_limit = []
    speed_direction_limit = []
    front_limit = []
    # Get onlt one value in one spike
    for t,v in enumerate(jump_times):
        time = jump_times[t]
        speed = deriv[t]

        if(energy[t]>threshold):
            value_limit.append(1)
        else:
            value_limit.append(0)

        if(speed>threshold_speed):
            speed_limit.append(1)
        else:
            speed_limit.append(0)

        if(speed>threshold_speed and speed_direction!=1):
            speed_direction = 1
        if(speed<-threshold_speed and speed_direction!=-1):
            speed_direction = -1
        if(speed>-threshold_speed and speed<threshold_speed and speed_direction!=0):
            speed_direction = 0

        speed_direction_limit.append(speed_direction)
        if(energy[t]>threshold and speed_direction==1):
            if(is_front!=1):
                is_front=1
                jump_x.append(time)
                jump_y.append(energy[t])
        elif (energy[t]>threshold and speed_direction==-1):
            if(is_front!=-1):
                is_front=-1
        else:
                is_front=0

        front_limit.append(is_front)

    if show_chart:
        # plt.figure(figsize=(10, 6)) 
        #fig, (ax1, ax2) = plt.subplots(2, 1)
        plt.title('Wave')
        librosa.display.waveshow(audio, sr=sample_rate)
        abs_window_size = 5        
        audio_absv = np.convolve(np.diff(np.abs(audio)), np.ones(abs_window_size)/abs_window_size, mode='valid')
        audio_absv = np.insert(  audio_absv,  0, np.zeros(abs_window_size//2))
        audio_absv = np.append(  audio_absv,   np.zeros(abs_window_size-abs_window_size//2))

        #plt.plot(jump_times[:len(deriv)], deriv)
        for i in range(len(jump_x)):
            t= jump_x[i]
            plt.plot([t,t], [1.,-1.], marker='o')
        
        plt.plot(jump_times, speed_limit, label="Speed")
        plt.plot(jump_times, value_limit, label="Energy")
        plt.plot(jump_times, speed_direction_limit,  label="Direction")
        plt.plot(jump_times, front_limit,  label="Front")
        
        plt.show()

        
        plt.title('Speed')
        plt.xlabel('Time')
        plt.plot(jump_times[:len(deriv)], deriv)
        plt.plot([0,jump_times[-1]], [threshold_speed,threshold_speed], marker='.')
        plt.plot([0,jump_times[-1]], [threshold_speed_mean,threshold_speed_mean], marker='.')
        plt.show()

        plt.title('Energy')
        plt.xlabel('Time')
        plt.plot(jump_times, energy)
        plt.plot(jump_x, jump_y, marker='o')
        plt.plot([0,jump_times[-1]], [threshold,threshold], marker='x')
        plt.plot([0,jump_times[-1]], [threshold_mean,threshold_mean], marker='.')

        #plt.xlim([0, len(energy)])
        #plt.xticks(np.arange(0, len(energy), step=50)) 
        plt.show()

    return (jump_x, jump_y)

