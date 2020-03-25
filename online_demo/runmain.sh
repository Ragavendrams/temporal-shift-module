#!/bin/sh
sudo find /var/log -type f -name "*.old" -exec rm -f {} \;
sudo find /var/log -type f -name "*.gz" -exec rm -f {} \;
sudo find /var/log -type f -name "*.1" -exec rm -f {} \;
sudo find . -name "*aarch64-linux*" -delete
trash-empty
sudo kill -9 $(pidof python3)
sudo service nvargus-daemon restart
python3 main.py 
