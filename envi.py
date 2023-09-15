import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import glob
import os

class Environment:

  def __init__(self):
    # self.data, self.rpm = self.get_random_csv_file()
    self.data, self.rpm = self.get_csv()
    self.threshold = 13
    self.window_size = 55
    self.step = 55
    self.state_step = 0
    self.peak = 0

    self.state_slide = self.sliding_window()

  def reset(self):
    self.data, self.rpm = self.get_random_csv_file()
    self.threshold = 13
    self.state_step = 0
    self.peak = 0

  def moving_average_filter(self, data, window):
    data = np.array(pd.Series(data).rolling(window = window).mean())
    return data

  def get_random_csv_file(self):
        # Lấy danh sách các tệp tin CSV trong thư mục hiện tại
        # csv_files = glob.glob('*.csv')

        folder_path = 'data/train'  # Tên của thư mục chứa các file CSV (data)
    
        # Lấy danh sách các tệp tin CSV trong thư mục đã chỉ định
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

        # Lấy ngẫu nhiên một tệp tin CSV
        random_csv_file = random.choice(csv_files)
        rpm = random_csv_file.split('-')[1].split('.')[0]
        rpm = int(rpm)

        # Đọc tệp tin CSV ngẫu nhiên
        data = pd.read_csv(random_csv_file, index_col=0)
        data = data.drop(data.index[0])

        data.iloc[:, -1] = data.iloc[:, -1].replace(';', '', regex=True)

        data = data.astype(float)

        # lọc trung bình bình phương
        data = (data['ax']**2 + data['ay']**2 + data['az']**2)**0.5
        data = np.array(data)
        data = data[0:2500]


        # lọc trung bình
        data = data.flatten() #Flatten a 2D numpy array into 1D array
        data = self.moving_average_filter(data, 50)

        # bỏ giá trị nan sau khi trượt cửa sổ trung bình
        data = data[~np.isnan(data)]

        return data, rpm

  # for testing purpose
  def get_csv(self):

        csv_file = os.path.join('data/train', '1-13.csv')
        rpm = csv_file.split('-')[1].split('.')[0]
        rpm = int(rpm)

        data = pd.read_csv(csv_file, index_col=0)
        data = data.drop(data.index[0])

        data.iloc[:, -1] = data.iloc[:, -1].replace(';', '', regex=True)

        data = data.astype(float)

        # lọc trung bình bình phương
        data = (data['ax']**2 + data['ay']**2 + data['az']**2)**0.5
        data = np.array(data)
        data = data[0:2500]


        # lọc trung bình
        data = data.flatten() #Flatten a 2D numpy array into 1D array
        data = self.moving_average_filter(data, 50)
        # data = data[0:3000]

        # bỏ giá trị nan sau khi trượt cửa sổ trung bình
        data = data[~np.isnan(data)]

        return data, rpm


  def sliding_window(self): #, window_size = 50, step = 40):
      window_size = self.window_size
      step = self.step
      result = []
      padding = window_size - 1
      padded_arr = np.pad(self.data, (padding, padding), mode='constant', constant_values=0)

      for i in range(0, len(padded_arr) - window_size + 1, step):
          window = padded_arr[i:i + window_size]
          result.append(window)

      return np.array(result)

  def get_state(self):
      state = self.state_slide[self.state_step]
      # print('total state: ', len(self.state_slide))
      # print('last state: ', state[-1])
      return state

  def step_action(self, action):
      done = False
      reward = 0
      # peak = 0
      loss = 0

      if action == 0:
          pass
      elif action == 1:
          self.peak += 1
          reward = 0.005
      else:
          print('invalid action')
      self.state_step += 1
      if self.state_step == len(self.state_slide):
        loss = abs(self.peak - self.rpm)
        if loss == 0:
            reward = 20

        elif loss == 1:
            reward = 5

        elif loss == 2:
            reward = -5

        elif loss == 3:
            reward = -10

        elif loss >= 4:
            reward = -20
        # reward = 1/(1+abs(self.peak-self.rpm))
        done = True
        self.reset()
        loss = loss ** 2


      return reward, done, loss
