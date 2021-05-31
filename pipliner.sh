#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name pipliner.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="pipliner.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

readonly RUN_TRAINING=$(cat ${JSON_FILE} | jq -r ".run_training")
readonly RUN_SEGMENTATION=$(cat ${JSON_FILE} | jq -r ".run_segmentation")
readonly RUN_CALUCULATION=$(cat ${JSON_FILE} | jq -r ".run_caluculation")

# Training input
readonly DATASET_MASK_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".dataset_mask_path"))
readonly DATASET_NONMASK_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".dataset_nonmask_path"))
save_directory="${DATASET_MASK_PATH}_nonmask/segmentation"

LOG_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".log_path"))

readonly IN_CHANNEL=$(cat ${JSON_FILE} | jq -r ".in_channel")
readonly NUM_CLASS=$(cat ${JSON_FILE} | jq -r ".num_class")
readonly LEARNING_RATE=$(cat ${JSON_FILE} | jq -r ".learning_rate")
readonly BATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".batch_size")
readonly DROPOUT=$(cat ${JSON_FILE} | jq -r ".dropout")
readonly NUM_WORKERS=$(cat ${JSON_FILE} | jq -r ".num_workers")
readonly EPOCH=$(cat ${JSON_FILE} | jq -r ".epoch")
readonly TRAIN_MASK_NONMASK_RATE=$(cat ${JSON_FILE} | jq -r ".train_mask_nonmask_rate")
readonly VAL_MASK_NONMASK_RATE=$(cat ${JSON_FILE} | jq -r ".val_mask_nonmask_rate")
readonly GPU_IDS=$(cat ${JSON_FILE} | jq -r ".gpu_ids")

# Segmentation input
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly MODEL_NAME=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_name"))

readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly AXIS=$(cat ${JSON_FILE} | jq -r ".axis")

readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")

readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly SAVE_NAME=$(cat ${JSON_FILE} | jq -r ".save_name")

# Caluculation input
readonly IGNORE_CLASSES=$(cat ${JSON_FILE} | jq -r ".ignore_classes")
readonly CSV_SAVEDIR=$(eval echo $(cat ${JSON_FILE} | jq -r ".csv_savedir"))
readonly CLASS_LABEL=$(cat ${JSON_FILE} | jq -r ".class_label")
readonly TRUE_NAME=$(cat ${JSON_FILE} | jq -r ".true_name")
readonly PREDICT_NAME=$(cat ${JSON_FILE} | jq -r ".predict_name")


readonly TRAIN_LISTS=$(cat ${JSON_FILE} | jq -r ".train_lists")
readonly VAL_LISTS=$(cat ${JSON_FILE} | jq -r ".val_lists")
readonly TEST_LISTS=$(cat ${JSON_FILE} | jq -r ".test_lists")
readonly KEYS=$(cat ${JSON_FILE} | jq -r ".train_lists | keys[]")

num_fold=(${KEYS// / })
num_fold=${#num_fold[@]}
LOG_PATH="${LOG_PATH}/${num_fold}-fold"

all_patients=""
for key in ${KEYS[@]}
do 
 echo $key
 TRAIN_LIST=$(echo $TRAIN_LISTS | jq -r ".$key")
 VAL_LIST=$(echo $VAL_LISTS | jq -r ".$key")
 TEST_LIST=$(echo $TEST_LISTS | jq -r ".$key")
 test_list=(${TEST_LIST// / })
 log_path="${LOG_PATH}/${key}"

 run_training_fold=$(echo $RUN_TRAINING | jq -r ".$key")
 run_segmentation_fold=$(echo $RUN_SEGMENTATION | jq -r ".$key")
 run_caluculation_fold=$(echo $RUN_CALUCULATION | jq -r ".$key")

 if ${run_training_fold};then
  echo "---------- Training ----------"

  python3 train.py ${DATASET_MASK_PATH} ${DATASET_NONMASK_PATH} ${log_path} --train_list ${TRAIN_LIST} --val_list ${VAL_LIST} --train_mask_nonmask_rate ${TRAIN_MASK_NONMASK_RATE} --val_mask_nonmask_rate ${VAL_MASK_NONMASK_RATE} --in_channel ${IN_CHANNEL} --num_class ${NUM_CLASS} --lr ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --epoch ${EPOCH} --gpu_ids ${GPU_IDS} --dropout ${DROPOUT}

   if [ $? -ne 0 ];then
    exit 1
   fi

 else
  echo "---------- No training ----------"
 fi

 model="${log_path}/${MODEL_NAME}"
 model_name=${model%.*}
 csv_name=${model_name////_}
 if ${run_segmentation_fold};then
  echo "---------- Segmentation ----------"
  for number in ${test_list[@]}
  do
   image="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
   save="${save_directory}/case_${number}/${SAVE_NAME}"

   if [ $MASK_NAME = "No" ];then
    mask=""

   else
    mask_path="${DATA_DIRECTORY}/case_${number}/${MASK_NAME}"
    mask="--mask_path ${mask_path}"
   fi

    python3 segmentation.py $image $model $save --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --overlap $OVERLAP -g ${GPU_IDS} ${mask} --num_class ${NUM_CLASS} --axis ${AXIS}

   if [ $? -ne 0 ];then
    exit 1
   fi

  done

 else
  echo "---------- No segmentation ----------"

 fi

 if ${run_caluculation_fold};then
  all_patients="${all_patients}${TEST_LIST} "
 fi
done

CSV_SAVEPATH="${CSV_SAVEDIR}/${csv_name}.csv"
echo "---------- Caluculation ----------"

python3 caluculateDICE.py ${DATA_DIRECTORY} ${save_directory} ${CSV_SAVEPATH} ${all_patients} --classes ${NUM_CLASS} --class_label ${CLASS_LABEL} --true_name ${TRUE_NAME} --predict_name ${PREDICT_NAME} --ignore_classes ${IGNORE_CLASSES}

if [ $? -ne 0 ];then
 exit 1
fi
