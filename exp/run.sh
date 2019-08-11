cudaN=2
des="ctb-conv-drop0.5"
exe="python ../../src/src-inside-outside-drop/main.py --exp-des $des --device cuda:$cudaN --config_file config.txt" 
#$exe --is_dictionary_exist 0 --is_train 1 --is_test 0 > log.create-dict 2>&1
# exit 0
# check the batch num for each iteration, before you run
$exe --is_dictionary_exist 1 --random_seed 1540422239 --is_train 1 --is_test 0 > log.train 2>&1

#exit 0
#num=`ls -d models-* | egrep -o '[0-9]+'`
#echo $num
#exe="python ../src-only-drop/main.py --exp-des $des --device cuda:$cudaN --config_file test-config.txt" 
#$exe --is_train 0 --is_test 1  > log.test 2>&1

#CUDA_VISIBLE_DEVICES=$cudaN $exe --is_train 0 --is_test 1 --model_eval_num $num #> log.test-$num 2>&1
