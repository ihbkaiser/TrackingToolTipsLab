# Improved Instance Segmentation-based Algorithm for Surgical Instrument Tip Detection
**[15.07.2024]** Paper submitted to [ICTA 2024](https://icta.hvu.edu.vn/). 

In this paper, I propose an algorithm based on segmentation masks to identify the tip point of a surgical instrument. We tested it on the Endovis 15 Dataset, achieving an accuracy of up to 87%.  

## Test
+ On the ground truth segmentation mask of Endovis15, we also added annotations of tip points using [CVAT](https://app.cvat.ai), represented by an XML file in each dataset.
+ Our dataset for segmentation problem is available [here](https://universe.roboflow.com/sugical-tool/sugical-tool-iszjm), you can use it to train your custom segmentation model.
+ Our trained segmentation models using the above dataset are `yolov9c-best.pt` and `yolov9e-best.pt`, you can download them [here](https://drive.google.com/drive/folders/1Z4_0maMJJLh1L1aYqa0nqFyKHb5ce_kB?usp=drive_link).
+ Just run the `ISI_v1.ipynb` to verify the results (we have tested with different segmentation models such as: GT Mask, `Roboflow-3-n-seg`, `YOLOv9c`, `YOLOv9e`.)  
## Demo
The yellow points are the tool tips detected by our algorithm, and the green points are the ground truth tips.
<p align="center">
  <img src="fig3.png" alt="Image 1" width="200"/>
  <img src="fig4.png" alt="Image 2" width="200"/>
  <img src="fig5.png" alt="Image 3" width="200"/>
</p>

We have uploaded a video to visualize our algorithm. You can view them [here](https://www.youtube.com/watch?v=RViEv6ap-dI).

