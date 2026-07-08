# pysph-example

```bash
conda env create -f env.yml
conda activate pysph

# Only if using OpenCL
ln -sf /etc/OpenCL/vendors/nvidia.icd "$CONDA_PREFIX/etc/OpenCL/vendors/nvidia.icd"
```

Example

```bash
python example.py
python example.py --openmp
python example.py --opencl
python example.py --cuda
python example.py --cuda --fused
```
