#!/bin/bash

REPO="path/to/repository/"
DATA="path/to/deepvariant_data/"
TRIALS_DIR="${1:-${REPO}current-run/}"


trial_dir="${TRIALS_DIR}training_validation" 
echo "Trial directory: $trial_dir"

test_dir="${TRIALS_DIR}testing"
sudo mkdir -p "$test_dir"
echo "Test dir created at: $test_dir"

ckpt_dir=$(find "$trial_dir/checkpoints" -maxdepth 1 -type d -name "ckpt-*" -print -quit)
echo "Checkpoint folder found: $ckpt_dir"

echo "Start DeepVariant"
docker run --rm \
   -v ${REPO}:${REPO} \
   -v ${DATA}:${DATA} \
   --gpus all \
   deepvariant/deepvariant_docker:v1 \
   run_deepvariant \
   --model_type WGS \
   --customized_model "$ckpt_dir" \
   --ref "${DATA}reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta" \
   --reads "${DATA}datasets/HG003.novaseq.pcr-free.30x.dedup.grch38.bam" \
   --output_vcf "$test_dir/test.vcf.gz" \
   --num_shards=128 \
   --logging_dir="$test_dir" \
   --regions "${DATA}reference/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent_chr21.bed" \

echo "Start hap.py"
docker run --rm -it \
   -v ${REPO}:${REPO} \
   -v ${DATA}:${DATA} \
   jmcdani20/hap.py:v0.3.12 \
   /opt/hap.py/bin/hap.py \
   "${DATA}reference/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
   "$test_dir/test.vcf.gz" \
   -f "${DATA}reference/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent_chr21.bed" \
   -r "${DATA}reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta" \
   -o "$test_dir/happy.output" \
   --engine=vcfeval \
   --engine-vcfeval-template "${DATA}reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.sdf" \
   --pass-only
