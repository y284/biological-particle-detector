# Model Card

### **Database Fields**

---

## Model Overview

**Model Name:** BPD

**Model Version:** *v1*

**Model Category**: *Segmentation*

**Repository:** [*https://github.com/y284/Fifth-place-solution*](https://github.com/y284/Fifth-place-solution)

**URI***: model URI*

**Release Date:** *Release date of the version currently available through public repository and described in this model card*

**Model Type and Purpose (“kind”):** *A U-Net based architecture for protein localization.*

**License:** *Apache 2.0*

**Model Developer:** *Youssef Ouertani*

**Model Description: *BPD*** *is a computer vision model designed to localize proteins in 3D cellular images. It is based on a **U-Net architecture** and trained on tomograms provided by the competition host **CZII**.*

### 

### **Markdown Fields**

---

## Model Details

**Model Architecture:**  The model uses a 3D U-Net architecture with 2 downsampling and 2 upsampling levels. Each level processes features through 28, 32, and 36 channels. Before each resolution change, a block of two 3D convolutions (each followed by BatchNorm and ReLU) extracts features, with trilinear interpolation handling upsampling and downsampling.

**Parameters:** *Number of parameters is 350K*

**Citation:** *Provide citation information for users of the model*

**Primary Contact Email:** *ouertaniyoussef@yahoo.fr*

**System requirements:** *The algorithm needs an **Nvidia GPU** and **CUDA** to run at reasonable speed (in particular for training). In my case it has been trained on a GPU P100. For running on other GPUs, some parameter values (e.g. patch and batch sizes) may need to be changed to adapt to available memory.*

## Intended Use

**Primary Use Cases:** 

* *Cell type classification*  
* *Protein localization*

***Out-of-Scope or Unauthorized Use Cases: Suggested language is provided below.***  
*Do not use the model for the following purposes:*

* *Use that violates applicable laws, regulations (including trade compliance laws), or third party rights such as privacy or intellectual property rights.*  
* *Any use that is prohibited by the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.*  
* *Any use that is prohibited by the [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy).*

## Training Details

**Training Data:** 7 tomograms and ground truth annotations for six protein complexes (Apo-ferritin, Beta-amylase, Beta-galactosidase, cytosolic ribosomes, thyroglobulin and VLP)

**Training Procedure:**   
The model was trained on 3D tomogram volumes with spherical labels (radius \= log2(given\_radius)\*0.8), normalized using min-max scaling based on averaged (5, 99\) percentiles across all 7 tomograms. Each epoch consisted of 1024 randomly sampled 128×128×128 patches (batches of 4\) with data augmentation including flipping, z-axis rotations (90°/180°/270°), and ±3% intensity shifts.   
Training ran for 35 epochs (4 hours total) using Adam (lr=0.0001, β₁=0.9, β₂=0.999) with *fp16 mixed precision*, gradient clipping, and label-smoothed cross-entropy (smoothing=0.01).

**Training Code:** *[Kaggle Link*](https://www.kaggle.com/code/youssefouertani/train-script-42-90)

**Data Sources**: [*Link to dataset*](https://cryoetdataportal.czscience.com/depositions/10310)

## Performance Metrics

**Metrics:** *The model was evaluated by calculating the F-beta metric with a beta value of 4\. The F-beta metric with a beta value of 4 is used to prioritize recall over precision, heavily penalizing missed particles while being more lenient on false positives. In this context, a particle is considered "true" if it lies within a factor of 0.5 of the particle of interest's radius. There are five particles of interest, with three "easy" particles (ribosome, virus-like particles, and apo-ferritin) assigned a weight of 1 and two "hard" particles (thyroglobulin and β-galactosidase) assigned a weight of 2\. The results are micro-averaged across multiple tomograms, ensuring that precision and recall are computed across the entire dataset before applying the F-beta formula. The higher beta value (4) and particle weights emphasize the correct identification of particles, particularly the "hard" ones, making recall the dominant factor in evaluating performance.*

**Test Dataset:** *[Public Test Data*](https://cryoetdataportal.czscience.com/datasets/10445?deposition-id=10310) *, [Private Test Data](https://cryoetdataportal.czscience.com/datasets/10446?deposition-id=10310)*

**Evaluation Results:** 

| *Public Score* | *Private Score* |
| :---- | :---- |
| *0.77982* | *0.78252*  |

## Biases, Risks, and Limitations

*This section identifies potential harms, misunderstandings, and technical and limitations. It also provides information on warnings and potential mitigations. **Suggestions are provided below.***

**Potential Biases:**

* *The model may reflect biases present in the training data.*

**Risks:**  
*Areas of risk may include but are not limited to:*

* *Inaccurate outputs or hallucinations*  
* *Incorrect prediction.*

**Limitations:**

* *The model's performance may be limited by the size of the training set.*

**Caveats and Recommendations:**

* *Review and validate outputs generated by the model.*  
* *We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when using the model.*


## Acknowledgements

*This research is supported by The CZ Imaging Institute, a Chan Zuckerberg Initiative Institute*.

## **Database Fields**

---

## Platform Resources to Link

*This section is a list of resources to link on the model card page. Below is a screenshot of where this information will be displayed on the model card page.*

**Associated Platform Resources:** *Please list tutorials, quickstarts, datasets, and other model cards that are related to this model.*  
**![][image1]**  


[image1]: <data:image/png;base64,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>