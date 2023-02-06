cd apex-master
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
# pip install inplace-abn
cd inplace_abn
python setup.py install
cd ..
pip install einops
pip install tensorboardX
pip install ipdb
pip install mmcv
pip install timm
pip install MulticoreTSNE