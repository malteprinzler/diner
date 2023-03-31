#!/bin/bash
cp venv.zip /tmp
unzip /tmp/venv.zip -d /tmp/
source /tmp/venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export MESA_SHADER_CACHE_DIR=/tmp/.cache

in_subject_nr=$1
out_subject_nr=$(printf "%03d" $in_subject_nr)
in_dir="data/FACESCAPE_RAW/$in_subject_nr"
out_dir="data/FACESCAPE_PROCESSED/$out_subject_nr"
python deps/facescape_preprocessing/process_dataset.py --dir_in $in_dir --dir_out $out_dir
