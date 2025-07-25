
## The structure of the repository 

Main Folder Structure:

```
We
├── _assets     
│   └── shapenet                         object database                 
├── _datasets							
│   ├── kitti                            kitti dataset
│   └── kitti_construct                  generate test cases
├── _queue_guided                        seed queue
├── system                               systems under test
├── blender                              blender script
├── config                               sensor and algorithm configuration
├── core                                 core module of We
├── third                                third-party repository
├── eval_tools                           tools for evaluation AP
├── init.py                              environment setup script
├── logger.py                            log                  
├── visual.py                            data visualisation scripts
└── demo.py                              quick start
```



## Installation

We implement and test the tool on a server with an Intel i7-10700K CPU (3.80 GHz), 48 GB RAM, and an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM). 

### Basic Dependency

Run the following command to install the dependencies



```bash
pip install -r requirements.txt
```

Set your project path `config.common_config.project_dir="YOUR/PROJECT/PATH"`

Before running the script, set the environment variable
`export PROJECT_DIR=/your/project/path`

### Quick Start

1. Install blender.

   We leverage blender, an open-source 3D computer graphics software, to build virtual
   camera sensor. 

   - install 4>blender>=3.3.1 from this [link](https://www.blender.org/download/)
   - setting the config `config.camera_config.blender_path="YOUR/BLENDER/PATH"`

2. Install S2CRNet **[optional]**.

   We leverage S2CRNet to improve the realism of the synthesized test cases.

   - download repo from [link](https://github.com/stefanLeong/S2CRNet) to `./third/S2CRNet` 

     `git clone git@github.com:stefanLeong/S2CRNet.git`

   - setting the config `config.camera_config.is_image_refine=True`

3. Install CENet **[optional]**.
   We leverage CENet to split road from point cloud and get accurate object positions. This step is optional because we provide the road label in demo KITTI dataset.

   - download repo from [link](https://github.com/huixiancheng/CENet) to `./third/CENet` 
     `git clone git@github.com:huixiancheng/CENet.git`

After installing all the necessary configurations, you can  run the `demo.py` file we provided to generate multi-modal data:

```bash
python init.py
python demo.py 
```

The result can be found at `./_datasets/kitti_construct/demo`. Then we can run `visual.py` to visualize the synthetic data

We can set the parameters to control the data generation.
```bash
python demo.py -select_size {1-len(dataset)} -modality {multi/pc/image}
```


## Complete Requirements

### Download Datasets 

1. KITTI
   - Download KITTI datasets from this [link](https://www.cvlibs.net/datasets/kitti/index.php) to `./_datasets/kitti`
2. ShapeNet
   - Download ShapeNet datasets from this [link](https://shapenet.org/) to `./_assets/shapnet`
   - Refer to this [link](https://github.com/CesiumGS/obj2gltf) to create a 3D model in gltf format.

### Generate Multi-modal data with Guidance

We can set the parameters to control the data generation.

```bash
python demo.py -select_size {1 - len(dataset)} -modality {multi/pc/image} -SYSTEM {$tag_name}
```

The result can be found at `./_datasets/kitti_construct/SYSTEM`.


### Visualization

We can use opencv/open3d to visualize image/point clouds.

### Evaluaaation

The generated data are organized in KITTI format. So with can use the generated data for evaluation easily.

      

## Custom Configuration

Run We on a custom dataset： 

1. Prepare dataset in KITTI dataset format.
2. Set your dataset path `config.common_config.kitti_dataset_root="YOUR/DATASET/PATH" `

Run We with custom 3D models：

1. Prepare your model files in gltf format. 
2. Set your model path `config.common_config.assets_dir ="YOUR/ASSETS/PATH"`
