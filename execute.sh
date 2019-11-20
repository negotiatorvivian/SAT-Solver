#!/bin/bash
#for test_recurrence_num in 50 70 100 120 150
#for test_recurrence_num in 150 180 200 220 250
for ((test_recurrence_num=175; test_recurrence_num<=250; test_recurrence_num+=5))

do
    for ((local_search_iteration=8500; local_search_iteration<=9500; local_search_iteration+=100))
#    for local_search_iteration in 3500 3600 3700 3800
    do
        for ((epsilon=45; epsilon<=55;epsilon+=1 ));do
            temp=$(echo "$epsilon")
#            echo $temp
            echo "$test_recurrence_num -b 1 -w $local_search_iteration"
            echo `/d/Anaconda3/envs/py35/python /d/zzw/PDP-Solver\/build\/scripts-3.5\/satyr.py /d/zzw/PDP-Solver/config\/Predict\/PDP-np-nd-np-gcnf-10-100-pytorch.yaml /d/zzw/PDP-Solver/datasets\/test\/sat $test_recurrence_num -b 1 -w $local_search_iteration>>sat20`
        done
    done
done