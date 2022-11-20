#!/bin/bash
# dataset/batchsize/lr/max_jump/epsilon_decay_steps/dropout

hidden=256
idx=1
start=1
end=10000

lr=.001000
while [ `expr $lr \> .0000001` -eq 1 ]
do
  batchsize=64
  while [ $batchsize -le 128 ]
  do
    dropout=.20
    while [ `expr .80 \> $dropout` -eq 1 ]
    do
      max_jump=3
      while [ $max_jump -le 5 ]
      do
        epsilon_decay_steps=100
        while [ $epsilon_decay_steps -le 500 ]
        do
          if [ $idx -ge $start ] && [ $idx -le $end ];then
            printf "idx:%d lr:%1.6f dropout:%1.2f batchsize:%d epsilon_decay_steps:%d max_jump:%d\n" \
            $idx $lr $dropout $batchsize $epsilon_decay_steps $max_jump
            
            CUDA_VISIBLE_DEVICES=1 python -u main.py --dataset rest14 --lr $lr --dropout $dropout --batchsize $batchsize --epsilon_decay_steps $epsilon_decay_steps --max_jump $max_jump \
             > ./out_rest14/out_${lr}_${dropout}_${batchsize}_${epsilon_decay_steps}_${max_jump}.out 2>&1
          fi

          idx=`expr $idx + 1`
          epsilon_decay_steps=`expr $epsilon_decay_steps + 50`
        done 
        
        max_jump=`expr $max_jump + 1`
      done
      
      dropout=`echo "scale=2; $dropout + .10" | bc`  #`echo "$dropout + .10"|bc`  
    done 
    
    batchsize=`expr $batchsize + 32`
  done

  lr=`echo "scale=6; $lr / 10" | bc` 
done

echo "finish"