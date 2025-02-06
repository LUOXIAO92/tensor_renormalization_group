#!/bin/bash

Dcut=64
K=6
b=40.96
Lx=6
Ly=6
eps=0

rgscheme="hotrg"

truncate=1e-10
degeps=0

gilteps=0

output_dir="./out_su2gauge/${rgscheme}_D${Dcut}_K${K}_gilt${gilteps}_deg${degeps}/${eps}/b${b}"
jobname="SU2gauge_${rgscheme}_D${Dcut}_K${K}_gilt${gilteps}_deg${degeps}_eps${eps}_b${beta}"

mkdir -p $output_dir

echo "start job"

mpiexec -np 4 python -u ./SU2_gauge.py \
    --dcut ${Dcut} \
    --K "(${K}, ${K}, ${K})" \
    --L "(${Lx}, ${Ly})" \
    --beta ${b}  \
    --eps ${eps} \
    --init_tensor_chunk     "(216, 6, 6)"\
    --coarse_graining_chunk "(64, 64, 1)"    \
    --degeneracy_eps ${degeps} \
    --truncate_eps ${truncate} \
    --rgscheme ${rgscheme} \
    --gilt_eps ${gilteps} \
    --Ngilt 1 \
    --Ncutlegs 2 \
    --out_dir ${output_dir} > ${output_dir}/${jobname}.log


#test