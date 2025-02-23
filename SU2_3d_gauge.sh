#!/bin/bash

np=1

Dcut=64
K=6
b=2.0
Lx=6
Ly=6
Lt=6
eps=0

rgscheme="atrg"

truncate=0
degeps=0

gilteps=0

output_dir="./out_3d_su2gauge/${rgscheme}_D${Dcut}_K${K}_gilt${gilteps}_deg${degeps}/${eps}/b${b}"
jobname="SU2gauge_${rgscheme}_D${Dcut}_K${K}_gilt${gilteps}_deg${degeps}_eps${eps}_b${b}_np${np}"

mkdir -p $output_dir

echo "start job"

mpiexec -np ${np} python -u ./SU2_gauge_3d.py \
    --dcut ${Dcut} \
    --K "(${K}, ${K}, ${K})" \
    --L "(${Lx}, ${Ly}, ${Lt})" \
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
    --out_dir ${output_dir} #> ${output_dir}/${jobname}.log


#test