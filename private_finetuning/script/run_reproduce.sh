CUDA_ID=0
EPSILON=-1
CLIP_NORM=1
BATCH_SIZE=32
MODEL=base

echo "Epsilon = 10 with Adaptive Clipping"

domians=("sports" "cloth" "mag_cn" "mag_us")
noises=(0.47 0.448 0.541 0.532)

for (( i=0; i<${#domians[@]}; i++ )); do
    domain=${domians[$i]}
    noise=${noises[$i]}
    echo "Running for domain: $domain with noise: $noise"
    bash lp_train_pvgalm_node_adaptive.sh $CUDA_ID $domain $EPSILON $noise $CLIP_NORM $BATCH_SIZE $MODEL
done

echo "Epsilon = 4 with Adaptive Clipping"

noises=(0.61 0.583 0.71 0.705)

for (( i=0; i<${#domians[@]}; i++ )); do
    domain=${domians[$i]}
    noise=${noises[$i]}
    echo "Running for domain: $domain with noise: $noise"
    bash lp_train_pvgalm_node_adaptive.sh $CUDA_ID $domain $EPSILON $noise $CLIP_NORM $BATCH_SIZE $MODEL
done

echo "Epsilon = 10 with Standard Clipping"

noises=(0.835 0.795 0.958 0.964)

for (( i=0; i<${#domians[@]}; i++ )); do
    domain=${domians[$i]}
    noise=${noises[$i]}
    echo "Running for domain: $domain with noise: $noise"
    bash lp_train_pvgalm_node_standard.sh $CUDA_ID $domain $EPSILON $noise $CLIP_NORM $BATCH_SIZE $MODEL
done

echo "Epsilon = 4 with Standard Clipping"

noises=(1.08 1.05 1.155 1.13)

for (( i=0; i<${#domians[@]}; i++ )); do
    domain=${domians[$i]}
    noise=${noises[$i]}
    echo "Running for domain: $domain with noise: $noise"
    bash lp_train_pvgalm_node_standard.sh $CUDA_ID $domain $EPSILON $noise $CLIP_NORM $BATCH_SIZE $MODEL
done