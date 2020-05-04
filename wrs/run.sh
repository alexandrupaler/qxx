#!/bin/bash
cd $1
export PYTHONPATH=$1:PYTHONPATH
source ./venv/bin/activate
python3 $2
