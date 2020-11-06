#!/bin/bash

# remove application folder
sudo rm -rf "/Applications/Python 3.8"

# remove library
sudo rm -rf "/Library/Frameworks/Python.framework"

# remove symbolic links
sudo rm -f "/usr/local/bin/2to3"*
sudo rm -f "/usr/local/bin/easy_install-"*
sudo rm -f "/usr/local/bin/idle3"*
sudo rm -f "/usr/local/bin/pip3"*
sudo rm -f "/usr/local/bin/pydoc3"*
sudo rm -f "/usr/local/bin/python3"*
