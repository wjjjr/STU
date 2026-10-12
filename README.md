# STU: Sparse Tensor Attention for Hyperspectral Unmixing

Reference implementation of **STU**, a self-supervised sparse tensor attention
framework for hyperspectral unmixing (HU).

STU starts from the mathematical connection between self-attention and Tucker decomposition.
Pairwise dependencies among local spatial–spectral cubes are encoded into a **sparse tensorized
attention core**, which is projected through mode-wise low-rank factor matrices to recover the
abundance representations. The abundances are constrained to be nonnegative and sum-to-one, and are
combined under the linear mixture model (LMM) to reconstruct the observed HSI.

Paper: *Sparse Tensor Attention for Hyperspectral Unmixing* —
J. Wang, F. Xiong, Z. Wen, Z. Zhao, J. Zhou, Y. Qian —
<https://ieeexplore.ieee.org/document/11693948>

## Repository layout

```
STU/
├── data/                               # datasets and the VCA initializations
│   ├── JasperRidge/                    
│   ├── Urban/                          
│   └── WashingtonDC/                   
├── experiment/
│   ├── stu_demo.py                     # entry point: data loading, training, evaluation
│   ├── utils.py                        # helper collection 
│   └── model/net.py                    # network
└── results/                            # outputs 
```

## Environment

- Python 3.11, PyTorch 2.5.1 (CUDA 12.1 build), `tltorch` 0.5.0 
- A CUDA GPU. 

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Quick start

1. Open `experiment/stu_demo.py` and set the dataset in the `__main__` block:

   ```python
   dataset = "JasperRidge"   # "JasperRidge" | "WashingtonDC" | "Urban"
   ```

2. Make sure `data_path` / `init_path` inside the same block point to the files under `data/`


3. Run:

   ```bash
   cd experiment
   python stu_demo.py
   ```

## Notes

- **Attention width.** In `model/net.py`, `self.dim2 = self.dim` sets the Q/K projection width equal
  to the patch feature dimension (13824), so the `dim2=...` argument passed by the demo has no
  effect. Use `self.dim2 = dim2` if you want the narrow projection instead.
- **Urban is large.** With 307 × 307 pixels the number of sliding patches, and hence the attention
  matrix, grows quickly; increase `stride` (e.g. `(12, 12, 12)`) if you run out of GPU memory.

## Citation

```bibtex
@article{wang2026stu,
  title   = {Sparse Tensor Attention for Hyperspectral Unmixing},
  author  = {Wang, Jingran and Xiong, Fengchao and Wen, Zhiyuan and Zhao, Zhuang and Zhou, Jun and Qian, Yuntao},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {\url{https://ieeexplore.ieee.org/document/11693948}}
}
```
