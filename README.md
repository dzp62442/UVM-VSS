<p align="center">
<h1 align="center"><strong> Unified Vertex Motion Estimation for Integrated Video Stabilization and Stitching in Tractor-Trailer Wheeled Robots</strong></h1>
</p>



<p align="center">
  <a href="https://inin-drops.github.io/UVM-VSS/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a> 
  
  <a href="https://arxiv.org/pdf/2412.07154" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 
  
  <a href="https://youtu.be/CqoVZQdvxU4" target='_blank'>
    <img src="https://img.shields.io/badge/Video-📹-red?">
  </a> 
</p>


 ## 🏠  Abstract
Tractor-trailer wheeled robots need to perform comprehensive perception tasks to enhance their operations in areas such as logistics parks and long-haul transportation. The perception of these robots face three major challenges: the relative pose change between the tractor and trailer, the asynchronous vibrations between the tractor and trailer, and the significant camera parallax caused by the large size. In this paper, we propose a novel Unified Vertex Motion Video Stabilization and Stitching framework designed for unknown environments. To establish the relationship between stabilization and stitching, the proposed Unified Vertex Motion framework comprises the Stitching Motion Field, which addresses relative positional change, and the Stabilization Motion Field, which tackles asynchronous vibrations. Then, recognizing the heterogeneity of optimization functions required for stabilization and stitching, a weighted cost function approach is proposed to address the problem of camera parallax. Furthermore, this framework has been successfully implemented in real tractor-trailer wheeled robots.  The proposed Unified Vertex Motion Video Stabilization and Stitching method has been thoroughly tested in various challenging scenarios, demonstrating its accuracy and practicality in real-world robot tasks.
<img src="https://github.com/lhlawrence/UVM-VSS/blob/main/poster.png">


## 🛠  Install

### Install the required libraries
Use conda to install the required environment. To avoid problems, it is recommended to follow the instructions below to set up the environment.


```bash
conda env create -f environment.yaml
conda activate uvm
```
### Clone this repo

```bash
git clone https://github.com/lhlawrence/UVM-VSS.git
cd UVM-VSS
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
  title={Unified Vertex Motion Estimation for Integrated Video Stabilization and Stitching in Tractor-Trailer Wheeled Robots}, 
  author={Hao Liang and Zhipeng Dong and Hao Li and Yufeng Yue and Mengyin Fu and Yi Yang},
  year={2024},
  eprint={2412.07154},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2412.07154}, 
}
```

## 👏 Acknowledgements
We would like to express our gratitude to All ININ members for their support and encouragement. 
