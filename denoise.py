from functools import total_ordering
import pandas as pd
import numpy as np
import sys
import requests
import os
import warnings
import time
warnings.filterwarnings("ignore")
from scipy.io import wavfile
from scipy.io.wavfile import write
import noisereduce as nr
import librosa


def gen_df(filename):

    df = pd.read_excel(filename)
    df1 = df.copy()
    df1 = df1[~df1['是否通过'].isin(['通过'])]
    df1 = df1[~df1['不完整'].isin([1])]
    df1 = df1[~df1['跳过歌词'].isin([1])]
    df1 = df1[~df1['跑调'].isin([1])]
    df1 = df1[~df1['唱错了歌词'].isin([1])]
    df1 = df1[~df1['没有声音'].isin([1])]
    df1 = df1[~df1['改旋律'].isin([1])]
    df1 = df1[~df1['多声音'].isin([1])]
    df1 = df1[~df1['资源有误'].isin([1])]
    df1 = df1[~df1['节奏不对'].isin([1])]
    df1 = df1.reset_index()
    new_df =  pd.DataFrame(df1, columns=['资源编号','URL'])
    return new_df



def download_url(url,id):
    resp = requests.get(url, timeout=300, verify=False)
    res = resp.content
    with open("songs/{}.m4a".format(id),"wb") as file:
        file.write(res)
        print("{}.m4a downloaded successfully".format(id))
    


def download(df):
    num = len(df['URL'])
    for i in range(num):
        if os.path.exists("songs/{}.m4a".format(df['资源编号'][i])):
            print("processing {}/{}, songs/{}.m4a is already been downloaded before".format(i, num, df['资源编号'][i]))
        else:
            print("processing {}/{}, songs/{}.m4a".format(i, num, df['资源编号'][i]))
            download_url(df['URL'][i], df['资源编号'][i])




def denoise(path):
    audio, sr = librosa.core.load(path, mono=True)
    audio_de = nr.reduce_noise(y = audio, sr=sr,stationary=True)
    audio_de = np.array(audio_de)
    new_audio = np.nan_to_num(audio_de, nan=0.0)
    return new_audio



def denoise_n_save(path, new_path):
    files = os.listdir(path)
    embd_list = []
    i = 1
    total_num = len(files)
    for file in files:
        file_size = os.path.getsize(path+'/'+file)
        if file_size <= 150000:
            print('processing {}/{}, id:{} size is too small, pass'.format(i, total_num, id))
        else: 
            start = time.time()
            if file[-3:] != 'm4a':  
                continue
            else:
                f = file.split('.')
                id = f[0]
                audio_de_array = denoise(path+'/'+file)
                write(new_path+"/{}.wav".format(id), 22050, audio_de_array)
                end = time.time()
                i += 1
                duration = end-start
                print('processing {}/{}, takes {}s to denoise id:{}'.format(i, total_num, duration, id, i))



def run(filename):
    df = gen_df(filename)
    df.to_excel('pruned_url.xls')
    download(df)
    denoise_n_save('songs', 'songs_denoised')



if __name__ == "__main__":
    file = sys.argv[1]
    run(file)
    