<p align="center">
  <h1 align="center">Multi-scale attention subtraction networks and an ultra-wide field retinal hemorrhage dataset</h1>
  <p align="center">
    Renkai Wu, Pengchen Liang, Yiqi Huang, Qing Chang* and Huiping Yao*
  </p>
    <p align="center">
      1. Ruijin Hospital, Shanghai Jiao Tong University School of Medicine, Shanghai, China</br>
      2. Shanghai University, Shanghai, China</br>
  </p>
</p>

## News🚀
(2024.10.10) ***The UWF-RHS dataset is now fully public.🔥*** 

(2024.09.17) ***Our model code has been uploaded! The UWF-RHS Dataset will provide access links next. Stay tuned!*** 

(2024.09.10) ***This work has been accepted for early access by IEEE Journal of Biomedical and Health Informatics!🔥*** 

(2024.03.02) ***Upload the corresponding running code.*** 

(2024.03.02) ***Manuscript submitted for review.*** 📃




**0. Main Environments.**
- python 3.8
- pytorch 1.12.0

**1. The proposed datasets (UWF-RHS).** </br>
The UWF-RHS dataset is available [here](https://drive.google.com/file/d/1gxNBujh8OrCyZ3cjtM59ze50VL-sR9qw/view?usp=sharing). It should be noted:
1. If you use the dataset, please cite the paper: https://doi.org/10.1109/JBHI.2024.3457512
2. The UWF-RHS dataset may only be used for academic research, not for commercial purposes.
3. If you can, please give us a like (Starred) for our GitHub project: https://github.com/wurenkai/UWF-RHS-Dataset-and-MASNet


https://github.com/wurenkai/UWF-RHS-Dataset-and-MASNet/assets/124028634/17a43a92-e421-4fae-bf61-1ce84b0eefca



**2. Train the MASNet.**
```
python train.py
```
- After trianing, you could obtain the outputs in './results/'

**3. Test the MASNet.** </br>
First, in the test.py file, you should change the address of the checkpoint in 'resume_model' and fill in the location of the test data in 'data_path'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/'

## Citation
If you find this repository helpful, please consider citing:
```
@article{wu2024automatic,
  title={Automatic Segmentation of Hemorrhages in the Ultra-wide Field Retina: Multi-scale Attention Subtraction Networks and An Ultra-wide Field Retinal Hemorrhage Dataset},
  author={Wu, Renkai and Liang, Pengchen and Huang, Yiqi and Chang, Qing and Yao, Huiping},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```
