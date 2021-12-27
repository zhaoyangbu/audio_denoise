import os
import subprocess

path = 'songs_denoised'
new_path = 'songs_denoised_m4a'
str = 'ffmpeg -i {} -c:a aac -b:a 128k -ac 2 {}'

for parent, dirnames, filenames in os.walk(path):
    for filename in filenames:
        old_dir = os.path.join(parent, filename)
        if old_dir[-3:] == 'wav':
            new_dir = new_path + '/' + filename.replace('.wav', '.m4a')
            str_cmd = str.format(old_dir, new_dir)
            print(str_cmd)

            p = subprocess.Popen(str_cmd, shell=True, stdout=subprocess.PIPE)
            for line in iter(p.stdout.readline, b''):
                print(line.strip().decode('gbk'))