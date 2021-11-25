# great-barrier-reef
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import glob
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 30, 30



df = pd.read_csv('../input/tensorflow-great-barrier-reef/train.csv')
df.head()



from ast import literal_eval


def load_image(video_id, video_frame, image_dir):
    img_path = f'{image_dir}/video_{video_id}/{video_frame}.jpg'
    assert os.path.exists(img_path), f'{img_path} does not exist.'
    img = cv2.imread(img_path)
    return img


def decode_annotations(annotaitons_str):
    """decode annotations in string to list of dict"""
    return literal_eval(annotaitons_str)

def load_image_with_annotations(video_id, video_frame, image_dir, annotaitons_str):
    img = load_image(video_id, video_frame, image_dir)
    annotations = decode_annotations(annotaitons_str)
    if len(annotations) > 0:
        for ann in annotations:
            cv2.rectangle(img, (ann['x'], ann['y']),
                (ann['x'] + ann['width'], ann['y'] + ann['height']),
                (0, 255, 255), thickness=2,)
    return img

#test
index = 16
row = df.iloc[index]
video_id = row.video_id
video_frame = row.video_frame
annotations_str = row.annotations
image_dir = '../input/tensorflow-great-barrier-reef/train_images'
img = load_image_with_annotations(video_id, video_frame, image_dir, annotations_str)
plt.imshow(img[:, :, ::-1])



from tqdm.auto import tqdm
import subprocess

def make_video(df, video_id, image_dir):
    # partly borrowed from https://github.com/RobMulla/helmet-assignment/blob/main/helmet_assignment/video.py
    fps = 15 # don't know exact value
    width = 1280
    height = 720
    save_path = f'video{video_id}.mp4'
    tmp_path = "tmp_" + save_path
    output_video = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))
    
    video_df = df.query('video_id == @video_id')
    for _, row in tqdm(video_df.iterrows(), total=len(video_df)):
        video_id = row.video_id
        video_frame = row.video_frame
        annotations_str = row.annotations
        img = load_image_with_annotations(video_id, video_frame, image_dir, annotations_str)
        output_video.write(img)
    
    output_video.release()
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(save_path):
        os.remove(save_path)
    subprocess.run(
        ["ffmpeg", "-i", tmp_path, "-crf", "18", "-preset", "veryfast", "-vcodec", "libx264", save_path]
    )
    os.remove(tmp_path)

for video_id in list(df['video_id'].unique()):
    make_video(df, video_id, image_dir)
    
    
    
from IPython.display import Video, display
Video('video0.mp4')
