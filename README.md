<p align="center">
<h1 align="center"><strong> Unified Vertex Motion Estimation for Integrated Video Stabilization and Stitching in Tractor-Trailer Wheeled Robots</strong></h1>
</p>



<p align="center">
  <a href="https://inin-drops.github.io/UVM-VSS/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a> 
  
  <a href="https://doi.org/10.1016/j.robot.2025.105004" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 

  <a href="https://arxiv.org/pdf/2412.07154" target='_blank'>
    <img src="https://img.shields.io/badge/Arxiv-📃-yellow?">
  </a> 
  
  <a href="https://youtu.be/CqoVZQdvxU4" target='_blank'>
    <img src="https://img.shields.io/badge/Video-📹-red?">
  </a> 
</p>


 ## 🏠  Abstract
Tractor–trailer wheeled robots need to perform comprehensive perception tasks to enhance their operations in areas such as logistics parks and long-haul transportation. The perception of these robots faces three major challenges: the asynchronous vibrations between the tractor and trailer, the relative pose change between the tractor and trailer, and the significant camera parallax caused by the large size. In this paper, we employ the Dual Independence Stabilization Motion Field Estimation method to address asynchronous vibrations between the tractor and trailer, effectively eliminating conflicting motion estimations for the same object in overlapping regions. We utilize the Random Plane-based Stitching Motion Field Estimation method to tackle the continuous relative pose changes caused by the articulated hitch between the tractor and trailer, thus eliminating dynamic misalignment in overlapping regions. Furthermore, we apply the Unified Vertex Motion Estimation method to manage the challenges posed by the tractor–trailer’s large physical size, which results in severely low overlapping regions between the tractor and trailer views, thus preventing distortions in overlapping regions from exponentially propagating into non-overlapping areas. Furthermore, this framework has been successfully implemented in real tractor–trailer wheeled robots. The proposed Unified Vertex Motion Video Stabilization and Stitching method has been thoroughly tested in various challenging scenarios, demonstrating its accuracy and practicality in real-world robot tasks.
<img src="https://github.com/lhlawrence/UVM-VSS/blob/main/poster.png">


## 🛠  Install

### Clone this repo

```bash
git clone https://github.com/lhlawrence/UVM-VSS.git
cd UVM-VSS
```

### Install the required libraries
Use conda to install the required environment. To avoid problems, it is recommended to follow the instructions below to set up the environment.


```bash
conda create -n uvm python=3.9
conda activate uvm
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 📊 Prepare dataset
Please download the following datasets.
* [UVM-VSS](https://huggingface.co/datasets/lhlawrence/UVM-VSS/resolve/main/data.zip) - unzip and place in root directory.

## 🏃 Run
### Main Code
Run the following command.

```bash
# for front view stitching
python front_view.py 
# for rear view stitching
python rear_view.py 
# for UVM-VSS
python stitch_dynamic.py 
```
You can see a visualization of the results in the ```data/final``` folder.
## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{UVM-VSS,
title = {Unified Vertex Motion Estimation for integrated video stabilization and stitching in tractor–trailer wheeled robots},
journal = {Robotics and Autonomous Systems},
volume = {191},
pages = {105004},
year = {2025},
issn = {0921-8890},
doi = {https://doi.org/10.1016/j.robot.2025.105004},
url = {https://www.sciencedirect.com/science/article/pii/S0921889025000909},
author = {Hao Liang and Zhipeng Dong and Hao Li and Yufeng Yue and Mengyin Fu and Yi Yang}
}
```

## 👏 Acknowledgements
We would like to express our gratitude to All ININ members for their support and encouragement. 
