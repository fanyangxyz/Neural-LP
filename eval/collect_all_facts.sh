folder=$1

pushd $folder
cat train.txt facts.txt valid.txt test.txt > all.txt
popd
