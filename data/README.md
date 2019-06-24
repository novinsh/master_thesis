This setup script prepares the synthetic and real datasets.
For the real dataset it downloads GEFCom2014 dataset from 
https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0, extracts
the Wind part of the dataset, transfers it to Pandas dataframes, and stores
it as pickle files for later usage. For more information refer to
http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html
