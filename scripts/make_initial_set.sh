#!/bin/bash

#train example
# find images.cv_tabbycat-noresize/data/train/ -type f -name '*.jpg' -print0 | head -z -n 14 | xargs -0 -r -- cp -t initial_set/train/no-squirrel/

#test example
# find images.cv_robin-noresize/data/test/ -type f -name '*.jpg' -print0 | head -z -n 3 | xargs -0 -r -- cp -t initial_set/test/no-squirrel/

# validation example
# find images.cv_tabbycat-noresize/data/val/ -type f -name '*.jpg' -print0 | head -z -n 3 | xargs -0 -r -- cp -t initial_set/validation/no-squirrel/

rm -fr initial_set
mkdir -p initial_set/{train,test,validation}/{squirrel,no-squirrel}

declare -A data_and_count
declare -A dir_mapper

# copy squirrel data
top_level_dir="images.cv_squirrels-noresize"
data_and_count["train"]=70
data_and_count["test"]=15
data_and_count["val"]=15

dir_mapper["train"]="train"
dir_mapper["test"]="test"
dir_mapper["val"]="validation"

animal_type="squirrel"

for dir in ${!dir_mapper[@]}; do
  mapped_dir=${dir_mapper[$dir]}
  search_dir="$top_level_dir/data/$mapped_dir"
  dest_dir="initial_set/$mapped_dir/$animal_type"
  file_count=${data_and_count[$dir]}

  find $search_dir -type f -name '*.jpg' -print0 | head -z -n $file_count | xargs -0 -r -- cp -t "$dest_dir"
done

# copy non-squirrel data
data_and_count["train"]=14
data_and_count["test"]=3
data_and_count["val"]=3

dir_mapper["train"]="train"
dir_mapper["test"]="test"
dir_mapper["val"]="validation"

animal_type="no-squirrel"

for top_level_dir in `find . -maxdepth 1 -not -path . -type d -name "images.cv*" -not -name "*squirrel*"`; do
  for dir in ${!dir_mapper[@]}; do
    mapped_dir=${dir_mapper[$dir]}
    search_dir="$top_level_dir/data/$dir"
    dest_dir="initial_set/$mapped_dir/$animal_type"
    file_count=${data_and_count[$dir]}

    find $search_dir -type f -name '*.jpg' -print0 | head -z -n $file_count | xargs -0 -r -- cp -t "$dest_dir"
  done
done
