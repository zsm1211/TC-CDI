# TC-CDI
Python(pytorch) code for the paper: Ziyang Chen, **Siming Zheng**, Zhishen Tong, and Xin Yuan, "Physics-driven deep learning enables temporal compressive coherent diffraction imaging," Optica, 9(6): 677–680[[pdf]](https://opg.optica.org/optica/viewmedia.cfm?uri=optica-9-6-677&seq=0) [[doi]](https://opg.optica.org/optica/viewmedia.cfm?uri=optica-9-6-677&seq=0)

## Abstract
Coherent diffraction imaging (CDI), as a lensless imaging technique, can achieve a high-resolution image with intensity and phase information from a diffraction pattern. To capture high speed and high spatial-resolution scenes, we propose a temporal compressive CDI system. A two-step algorithm using physics-driven deep-learning networks is developed for multi-frame spectra reconstruction and phase retrieval. Experimental results demonstrate that our system can reconstruct up to 8(20) frames from a snapshot measurement. Our results offer huge potential for visualizing the dynamic process of molecules with large field-of-view, high spatial and temporal resolutions.

<p align="left">
<img src="https://github.com/zsm1211/TC-CDI/blob/main/TC_CDI_result.png?height="600" width="1000"raw=true">
</p>

Figure 1.Reconstruction results for the complicated object. (a) Coded measurement; (b) Reference images of the moving object; (c) reconstructed spatial spectra; (d) 8 corresponding reconstructed spatial images by HIO algorithm; (e) 8 corresponding reconstructed spatial images by the proposed DNN-HIO algorithm. Boxes of different colors circle the parts where the contrast between the two results is more obvious.

## Usage

1.Download the all the files via [Baidu Drive](https://pan.baidu.com/s/1_vzOj8CFbyLGHY7tEYt5YA) (access code `zsms`) or [One Drive](https://westlakeu-my.sharepoint.com/:f:/g/personal/xylab_westlake_edu_cn/EijpeAUWWShHqipaqwfR084BS40MS_edJ7GwNJs0ks2TEg?e=wmEZ8b) and directly put the data in `TC_CDI_Stage1`.

## Citation
```
@article{Chen:22,
author = {Ziyang Chen and Siming Zheng and Zhishen Tong and Xin Yuan},
journal = {Optica},
keywords = {Digital micromirror devices; Phase retrieval; Power spectral density; Ptychography; Spatial resolution; X ray imaging},
number = {6},
pages = {677--680},
publisher = {Optica Publishing Group},
title = {Physics-driven deep learning enables temporal compressive coherent diffraction imaging},
volume = {9},
month = {Jun},
year = {2022},
url = {http://opg.optica.org/optica/abstract.cfm?URI=optica-9-6-677},
doi = {10.1364/OPTICA.454582},
abstract = {Coherent diffraction imaging (CDI), as a lensless imaging technique, can achieve a high-resolution image with intensity and phase information from a diffraction pattern. To capture high-speed and high-spatial-resolution scenes, we propose a temporal compressive CDI system. A two-step algorithm using physics-driven deep-learning networks is developed for multi-frame spectra reconstruction and phase retrieval. Experimental results demonstrate that our system can reconstruct up to eight frames from a snapshot measurement. Our results offer the potential to visualize the dynamic process of molecules with large fields of view and high spatial and temporal resolutions.},
}
```
