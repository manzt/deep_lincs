#!/bin/bash

ACC=GSE92742
PREFIX=${ACC}_Broad_LINCS
GEO_URL=https://www.ncbi.nlm.nih.gov/geo

files=(
    Level3_INF_mlr12k_n1319138x12328.gctx
    cell_info.txt
    gene_info.txt
    inst_info.txt
    pert_info.txt
)

mkdir $PREFIX

for f in ${files[*]}; do
  curl "$GEO_URL/download/?acc=$ACC&format=file&file=${PREFIX}_$f.gz" \
    | gunzip > $PREFIX/$f
done

# remove "pr_" from gene metadata cols
sed -i "s/pr_//g" $PREFIX/gene_info.txt
