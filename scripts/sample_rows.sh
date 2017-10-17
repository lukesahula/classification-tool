counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_1000/010_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 100 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_1000/sampled)"
    counter=$((counter+100))
    echo "$counter / 1000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_100000/050_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_100000/sampled)"
    counter=$((counter+2000))
    echo "$counter / 100000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_200000/100_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_200000/sampled)"
    counter=$((counter+2000))
    echo "$counter / 200000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_400000/200_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_samples_400000/sampled)"
    counter=$((counter+2000))
    echo "$counter / 400000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_1000/010_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170111/neg/"$zip" | shuf -n 100 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_1000/sampled)"
    counter=$((counter+100))
    echo "$counter / 1000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_100000/050_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170111/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_100000/sampled)"
    counter=$((counter+2000))
    echo "$counter / 100000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_200000/100_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170111/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_200000/sampled)"
    counter=$((counter+2000))
    echo "$counter / 200000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_400000/200_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170111/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170111/neg_seed_01_samples_400000/sampled)"
    counter=$((counter+2000))
    echo "$counter / 400000 rows sampled."
done
