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
(2024.09.17) ***Our model code has been uploaded! The UWF-RHS Dataset will provide access links next. Stay tuned!*** 

(2024.09.10) ***This work has been accepted for early access by IEEE Journal of Biomedical and Health Informatics!🔥*** 

(2024.03.02) ***Upload the corresponding running code.*** 

(2024.03.02) ***Manuscript submitted for review.*** 📃

***Note: The UWF-RHS Dataset be live soon! Keep following us!***

**0. Main Environments.**
- python 3.8
- pytorch 1.12.0

**1. The proposed datasets (UWF-RHS).**
- Given the value of collecting and annotating data, we would prefer to provide access links after the paper has been accepted. However, for ease of review, we will provide some examples. Your understanding is greatly appreciated.



https://github.com/wurenkai/UWF-RHS-Dataset-and-MASNet/assets/124028634/17a43a92-e421-4fae-bf61-1ce84b0eefca



**2. Train the MASNet.**
```
python train.py
```
- After trianing, you could obtain the outputs in './results/'

**3. Test the MHorUNet.**</br>
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
