#!/bin/bash
#ENVIRONMENT ARRAY
declare -a envs=("Pong-v0" "" "");
#START BATCH SIZE LOOP
declare -a bSize=();
#this is the amount to increase each iteration by
bAcc=20;
bAdd=20;
#this is the amount of times to run the file
declare -i bLen=5;
for i in $(seq 0 $bLen);
do
bSize+=($bAcc);
 ((bAcc+=$bAdd))
done
#START EPISODE LOOP
declare -a eps=();
#this is the amount to increase each iteration by
eAcc=100;
eAdd=10;
#this is the amount of times to run the file
declare -i eLen=100;
for i in $(seq 0 $eLen);
do
eps+=($eAcc);
 ((eAcc+=$eAdd))
done
#START LEARNING RATE LOOP
declare -a lRates=();

#this is the amount of times to run the file
declare -i lLen=5;
for i in $(seq 2 $lLen);
do
lAcc=$(awk -v i="$i" 'BEGIN{a=.1^i;printf("%0.*f\n" ,i , a)}')
lRates+=("$lAcc");
done
#TRAINING LOOP
for b in ${bSize[@]}
do
  for m in ${eps[@]}
  do
    for e in ${envs[@]}
    do
      for l in ${lRates[@]}
      do
        python trainer.py -b $b -e $e -m $m -l $l
      done
    done
  done
done
#END FILE

