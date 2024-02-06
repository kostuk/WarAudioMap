import os
import librosa
import numpy as np
import pandas as pd

# Создайте пустой DataFrame для хранения результатов
results_df = pd.DataFrame(columns=['file', 'pick', 'power'])

# Путь к папке, содержащей ваши WAV-файлы
folder_path = 'raw_data/pushka'
# Specify the desired sampling rate (e.g., 44100 Hz)
desired_sr = 44100

# Проход по всем файлам в папке
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                # Resample the audio to the desired sampling rate
                if sr!=desired_sr :
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=desired_sr)
                    sr=desired_sr

                # audio = audio[:sr]  # Обрезка до 1 секунды
                
                # Вычисление максимальной энергии и времени её возникновения
                energy = np.sum(np.square(audio))
                max_energy_time = np.argmax(np.square(audio)) / sr
                duration = librosa.get_duration(y=audio, sr=sr)
                
                # Добавление результатов в DataFrame
                print({'file': file, 'pick': max_energy_time,'power':energy, 'sr':sr, 'duration':duration})
                #results_df = results_df.append({'file': file, 'pick': max_energy_time,'power':energy}, ignore_index=True)
                
            except Exception as e:
                print(f"Ошибка при обработке файла {file}: {str(e)}")

# Сохранение результатов в Excel-файл
output_excel_path = 'pushka.xlsx'
results_df.to_excel(output_excel_path, index=False, engine='openpyxl')
print(f"Результаты сохранены в {output_excel_path}")
