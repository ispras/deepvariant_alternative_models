#!/bin/bash

REPO="/repo/path/"
DATA="/data/path/"
DIR="results_dir_name" # "nasnetlarge-random-h1", "resnet152v2-random-h1", "efficientnetb03-random-h1", "inception-random"
CKPT="ckpt" #"ckpt-515000", "ckpt-518754", "ckpt-455000", "ckpt-518754"

docker run --rm \
   -v ${REPO}:${REPO} \
   -v ${DATA}:${DATA} \
   --gpus all \
   deepvariant/deepvariant_docker:v1 \
   run_deepvariant \
   --model_type WGS \
   --customized_model "${REPO}experiment/results/${DIR}/checkpoints/${CKPT}" \
   --ref "${DATA}reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta" \
   --reads "${DATA}datasets/HG003.novaseq.pcr-free.30x.dedup.grch38.bam" \
   --output_vcf "${REPO}experiment/results_test/${DIR}_20-22/test.vcf.gz" \
   --num_shards=128 \
   --logging_dir="${REPO}experiment/results_test/${DIR}_20-22/" \
   --regions "${DATA}reference/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent_chr20-22.bed"

docker run --rm -it \
    -v ${REPO}:${REPO} \
    -v ${DATA}:${DATA} \
    jmcdani20/hap.py:v0.3.12 \
    /opt/hap.py/bin/hap.py \
    "${DATA}reference/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" \
    "${REPO}experiment/results_test/${DIR}_20-22/test.vcf.gz" \
    -f "${DATA}reference/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent_chr20-22.bed" \
    -r "${DATA}reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta" \
    -o "${REPO}experiment/results_test/${DIR}_20-22/happy.output" \
    --engine=vcfeval \
    --engine-vcfeval-template "${DATA}reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.sdf" \
    --pass-only