#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
dir_path = 'GEFCom2014-W_V2/Wind/'
for t in range(1,16): # nr tasks
    ###################################
    df_power = []
    for i in range(1,11):
        path = dir_path+'Task {0}/Task{0}_W_Zone1_10/Task{0}_W_Zone{1}.csv'.format(t, i)
        #print(path)
        df_power_zone = pd.read_csv(path, 
                                    header=0, 
                                    usecols=[1, 2], 
                                    names=['datetime', 'wf'+str(i)])
        df_power_zone['datetime'] = pd.to_datetime(df_power_zone['datetime'], format='%Y%m%d %H:%M')
        df_power_zone.index = df_power_zone['datetime']
        df_power_zone = df_power_zone.drop(['datetime'], axis=1)
        df_power.append(df_power_zone)
    to_save_path = "./" + "/".join(d for d in path.split('/')[:-2]) + "/"
    df_power = pd.concat(df_power, axis=1, join='outer')
    df_power.to_pickle(to_save_path+"df_power.pkl")
    print(to_save_path)
    print(" power dates:")
    print("  ", df_power.index[0])
    print("  ", df_power.index[-1])
    ###################################
    df_wind = []    
    for i in range(1,11):
        path = dir_path+'Task {0}/Task{0}_W_Zone1_10/Task{0}_W_Zone{1}.csv'.format(t,i)
        #print(path)
        df_wind_zone = pd.read_csv(path, 
                              header=0, 
                              usecols=[1, 3, 4, 5, 6],
                              names=['datetime', 'U10', 'V10', 'U100', 'V100'])
        df_wind_zone['datetime'] = pd.to_datetime(df_wind_zone['datetime'], format='%Y%m%d %H:%M')
        df_wind_zone.index = df_wind_zone['datetime']
        df_wind_zone = df_wind_zone.drop(['datetime'], axis=1)
        df_wind.append(df_wind_zone)
    df_wind = pd.concat(df_wind, axis=1, keys=[str(i) for i in range(1,11)])
    df_wind.to_pickle(to_save_path+"df_wind.pkl")
    print(" wind dates:")
    print("  ", df_wind.index[0])
    print("  ", df_wind.index[-1])
    ###################################
    df_bench = []
    df_bench_task = pd.read_csv(dir_path+'/Task {0}/benchmark{0}_W.csv'.format(t), 
                                header=0,
                                usecols=range(101),
                                names=['wf', 'datetime']+[j for j in range(1,100)])
    df_bench_task['datetime'] = pd.to_datetime(df_bench_task['datetime'], format='%Y%m%d %H:%M')
    df_bench_task.index = df_bench_task['datetime']
    df_bench_task = df_bench_task.drop(['datetime'], axis=1)
    df_bench_task = df_bench_task.pivot(columns='wf')
    df_bench_task = df_bench_task.swaplevel(i=0, j=1, axis=1)
    df_bench.append(df_bench_task)
    df_bench = pd.concat(df_bench, axis=0, join='outer')
    df_bench.to_pickle(to_save_path+"df_bench.pkl")
    print(" benchmark dates:")
    print("  ", df_bench.index[0])
    print("  ", df_bench.index[-1])
    print("#")

