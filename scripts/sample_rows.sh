counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_sampled_050/050_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_sampled_050/sampled_100000)"
    counter=$((counter+2000))
    echo "$counter / 100000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_sampled_100/100_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_sampled_100/sampled_200000)"
    counter=$((counter+2000))
    echo "$counter / 200000 rows sampled."
done

counter=0
for zip in $(cat ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_sampled_200/200_negs)
do
    echo "$(zcat ../classification_tool/datasets/cisco_datasets/data/20170104/neg/"$zip" | shuf -n 2000 --random-source=seed_01 >> ../classification_tool/datasets/cisco_datasets/data/20170104/neg_seed_01_sampled_200/sampled_400000)"
    counter=$((counter+2000))
    echo "$counter / 400000 rows sampled."
done
