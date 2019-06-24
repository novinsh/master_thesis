#!/bin/bash

fold -w 80 -s README.md
read -rsp $'Press any key to begin...\n' -n1 key

# download the GEFCom2014 data
wget -O GEFCom2014.zip "https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1"

## unzip
unzip GEFCom2014.zip
unzip GEFCom2014\ Data/GEFCom2014-W_V2.zip -d GEFCom2014-W_V2
rm GEFCom2014\ Data -rf

# extract all .zip files recursively in their own directory
find GEFCom2014-W_V2 -name '*.zip' -exec sh -c 'unzip -d "$(dirname "$1")" "$1"' _ {} \;

# create pkl files
./csv_pickle.py

# remove all the unziped files recursively
find GEFCom2014-W_V2 -name '*.zip' -exec sh -c 'rm -rvf "${1%.*}"' _ {} \;

# remove all the zip files as well
find GEFCom2014-W_V2 -name '*.zip' -exec sh -c 'rm -rvf "${1}"' _ {} \;

# remove all csv files
find GEFCom2014-W_V2 -name '*.csv' -exec sh -c 'rm -rvf "${1}"' _ {} \;

rm GEFCom2014.zip

echo "Done. Let the fun begin :)"
