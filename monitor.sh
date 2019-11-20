#!/bin/bash
while True
do
	res=$(wmic process get commandline|findstr "satyr-train-test.py"|findstr -v "findstr.exe"|tr [:blank:] [a])

	echo "$res"
	if [ $res ]
	then
	   echo "包含"
	   sleep 20m
	else
	   echo "不包含"
	   echo `bash /d/zzw/PDP-Solver/run.sh`
	   # echo `/d/Anaconda3/envs/py35/python /d/zzw/PDP-Solver/build/scripts-3.5/satyr-train-test.py -l last /d/zzw/PDP-Solver/config/Train/p-prodec2-modular-variable-pytorch.yaml`
	   # sleep 5m
	fi
done