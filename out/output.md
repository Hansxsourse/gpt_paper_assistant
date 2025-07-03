# Personalized Daily ArXiv Papers 07/03/2025
Total relevant papers: 4

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](#link0)
**Authors:** Yuanrui Wang, Cong Han, Yafei Li, Zhipeng Jin, Xiawei Li, SiNan Du, Wen Tao, Yi Yang, Shuanglong Li, Chun Yuan, Liu Lin

1. [DiGA3D: Coarse-to-Fine Diffusional Propagation of Geometry and Appearance for Versatile 3D Inpainting](#link1)
**Authors:** Jingyi Pan, Dan Xu, Qiong Luo

2. [LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](#link2)
**Authors:** Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, Xiangxiang Chu

3. [PlantSegNeRF: A few-shot, cross-dataset method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching](#link3)
**Authors:** Xin Yang (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Ruiming Du (Department of Biological and Environmental Engineering, Cornell University), Hanyang Huang (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Jiayang Xie (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Pengyao Xie (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Leisen Fang (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Ziyue Guo (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Nanjun Jiang (Amway), Yu Jiang (Horticulture Section, School of Integrative Plant Science, Cornell AgriTech), Haiyan Cen (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs)

---
## 0. [UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](https://arxiv.org/abs/2507.00992) <a id="link0"></a>
**ArXiv ID:** 2507.00992
**Authors:** Yuanrui Wang, Cong Han, Yafei Li, Zhipeng Jin, Xiawei Li, SiNan Du, Wen Tao, Yi Yang, Shuanglong Li, Chun Yuan, Liu Lin

**Abstract:**  Text-to-image generation has greatly advanced content creation, yet accurately rendering visual text remains a key challenge due to blurred glyphs, semantic drift, and limited style control. Existing methods often rely on pre-rendered glyph images as conditions, but these struggle to retain original font styles and color cues, necessitating complex multi-branch designs that increase model overhead and reduce flexibility. To address these issues, we propose a segmentation-guided framework that uses pixel-level visual text masks -- rich in glyph shape, color, and spatial detail -- as unified conditional inputs. Our method introduces two core components: (1) a fine-tuned bilingual segmentation model for precise text mask extraction, and (2) a streamlined diffusion model augmented with adaptive glyph conditioning and a region-specific loss to preserve textual fidelity in both content and style. Our approach achieves state-of-the-art performance on the AnyText benchmark, significantly surpassing prior methods in both Chinese and English settings. To enable more rigorous evaluation, we also introduce two new benchmarks: GlyphMM-benchmark for testing layout and glyph consistency in complex typesetting, and MiniText-benchmark for assessing generation quality in small-scale text regions. Experimental results show that our model outperforms existing methods by a large margin in both scenarios, particularly excelling at small text rendering and complex layout preservation, validating its strong generalization and deployment readiness.

**Comment:** Matches criteria 1 and 2 closely with a unified segmentation-conditioned diffusion model for text synthesis.
**Relevance:** 10
**Novelty:** 8

---

## 1. [DiGA3D: Coarse-to-Fine Diffusional Propagation of Geometry and Appearance for Versatile 3D Inpainting](https://arxiv.org/abs/2507.00429) <a id="link1"></a>
**ArXiv ID:** 2507.00429
**Authors:** Jingyi Pan, Dan Xu, Qiong Luo

**Abstract:**  Developing a unified pipeline that enables users to remove, re-texture, or replace objects in a versatile manner is crucial for text-guided 3D inpainting. However, there are still challenges in performing multiple 3D inpainting tasks within a unified framework: 1) Single reference inpainting methods lack robustness when dealing with views that are far from the reference view. 2) Appearance inconsistency arises when independently inpainting multi-view images with 2D diffusion priors; 3) Geometry inconsistency limits performance when there are significant geometric changes in the inpainting regions. To tackle these challenges, we introduce DiGA3D, a novel and versatile 3D inpainting pipeline that leverages diffusion models to propagate consistent appearance and geometry in a coarse-to-fine manner. First, DiGA3D develops a robust strategy for selecting multiple reference views to reduce errors during propagation. Next, DiGA3D designs an Attention Feature Propagation (AFP) mechanism that propagates attention features from the selected reference views to other views via diffusion models to maintain appearance consistency. Furthermore, DiGA3D introduces a Texture-Geometry Score Distillation Sampling (TG-SDS) loss to further improve the geometric consistency of inpainted 3D scenes. Extensive experiments on multiple 3D inpainting tasks demonstrate the effectiveness of our method. The project page is available at https://rorisis.github.io/DiGA3D/.

**Comment:** Matches criteria 2 with a unified diffusion model for 3D inpainting tasks.
**Relevance:** 5
**Novelty:** 7

---

## 2. [LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](https://arxiv.org/abs/2507.00790) <a id="link2"></a>
**ArXiv ID:** 2507.00790
**Authors:** Huaqiu Li, Yong Wang, Tongwen Huang, Hailang Huang, Haoqian Wang, Xiangxiang Chu

**Abstract:**  Unified image restoration is a significantly challenging task in low-level vision. Existing methods either make tailored designs for specific tasks, limiting their generalizability across various types of degradation, or rely on training with paired datasets, thereby suffering from closed-set constraints. To address these issues, we propose a novel, dataset-free, and unified approach through recurrent posterior sampling utilizing a pretrained latent diffusion model. Our method incorporates the multimodal understanding model to provide sematic priors for the generative model under a task-blind condition. Furthermore, it utilizes a lightweight module to align the degraded input with the generated preference of the diffusion model, and employs recurrent refinement for posterior sampling. Extensive experiments demonstrate that our method outperforms state-of-the-art methods, validating its effectiveness and robustness. Our code and data will be available at https://github.com/AMAP-ML/LD-RPS.

**Comment:** 2
**Relevance:** 5
**Novelty:** 5

---

## 3. [PlantSegNeRF: A few-shot, cross-dataset method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching](https://arxiv.org/abs/2507.00371) <a id="link3"></a>
**ArXiv ID:** 2507.00371
**Authors:** Xin Yang (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Ruiming Du (Department of Biological and Environmental Engineering, Cornell University), Hanyang Huang (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Jiayang Xie (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Pengyao Xie (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Leisen Fang (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Ziyue Guo (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs), Nanjun Jiang (Amway), Yu Jiang (Horticulture Section, School of Integrative Plant Science, Cornell AgriTech), Haiyan Cen (College of Biosystems Engineering and Food Science, Zhejiang University, Key Laboratory of Spectroscopy Sensing, Ministry of Agriculture and Rural Affairs)

**Abstract:**  Organ segmentation of plant point clouds is a prerequisite for the high-resolution and accurate extraction of organ-level phenotypic traits. Although the fast development of deep learning has boosted much research on segmentation of plant point clouds, the existing techniques for organ segmentation still face limitations in resolution, segmentation accuracy, and generalizability across various plant species. In this study, we proposed a novel approach called plant segmentation neural radiance fields (PlantSegNeRF), aiming to directly generate high-precision instance point clouds from multi-view RGB image sequences for a wide range of plant species. PlantSegNeRF performed 2D instance segmentation on the multi-view images to generate instance masks for each organ with a corresponding ID. The multi-view instance IDs corresponding to the same plant organ were then matched and refined using a specially designed instance matching module. The instance NeRF was developed to render an implicit scene, containing color, density, semantic and instance information. The implicit scene was ultimately converted into high-precision plant instance point clouds based on the volume density. The results proved that in semantic segmentation of point clouds, PlantSegNeRF outperformed the commonly used methods, demonstrating an average improvement of 16.1%, 18.3%, 17.8%, and 24.2% in precision, recall, F1-score, and IoU compared to the second-best results on structurally complex datasets. More importantly, PlantSegNeRF exhibited significant advantages in plant point cloud instance segmentation tasks. Across all plant datasets, it achieved average improvements of 11.7%, 38.2%, 32.2% and 25.3% in mPrec, mRec, mCov, mWCov, respectively. This study extends the organ-level plant phenotyping and provides a high-throughput way to supply high-quality 3D data for the development of large-scale models in plant science.

**Comment:** The paper does not match any of the specific criteria closely. It focuses on plant 3D instance point cloud reconstruction using a novel approach called PlantSegNeRF, which involves multi-view image instance matching and rendering implicit scenes. This does not align with the criteria of unified image/video generation and segmentation, unified diffusion models, image matting, or video matting.
**Relevance:** 3
**Novelty:** 6

---


---

## Paper selection prompt
Unified Image/Video Generation and Segmentation

Relevant: Papers that propose architectures or frameworks where image (or video) generation and semantic (or instance) segmentation are learned jointly or in a unified pipeline. Typically these works will explicitly describe a single model (e.g., a GAN, VAE, diffusion network, or transformer) that outputs both RGB pixels and segmentation maps for images or frames, or that uses segmentation information to guide generation. Look for titles or abstracts mentioning “joint generation and segmentation,” “multi-task generative segmentation,” “co-learning of synthesis and masks,” or “segmentation-aware generation.”

Not relevant: Papers that address generation or segmentation in isolation (e.g., a standard GAN paper that does not incorporate segmentation, or a segmentation network that does not produce novel images). Also exclude works where segmentation is merely an auxiliary loss without producing a full segmentation map (e.g., classification-based saliency or attention modules that do not yield a full semantic mask).

Unified Diffusion Models (Multi-Task: Low-Level Vision, Image Generation, Segmentation, Depth Estimation, Surface Normals, etc.)

Relevant: Papers that introduce diffusion (or score-based) models designed to handle multiple vision tasks under a single architecture or training regime. These should explicitly mention training a diffusion backbone (or shared denoiser) and then switching heads (or prompts) to perform tasks such as image denoising, super-resolution, inpainting, unconditional/conditional generation, semantic segmentation, monocular depth estimation, or surface normal prediction. Emphasis is on works that present a unified denoising framework (e.g., one U-Net or transformer backbone) with modular output branches or conditioning mechanisms for each task.

Not relevant: Diffusion papers that focus exclusively on one task (e.g., super-resolution only, or generation only). Also exclude works on diffusion in non-vision domains (e.g., text or audio), or papers that merely compare diffusion against other methods without proposing a multi-task, shared-diffusion backbone. If a paper briefly mentions a secondary task without truly integrating it into a unified training objective, it should be omitted.

Image Matting

Relevant: Papers that specifically target the problem of predicting a high-quality alpha matte for foreground extraction in still images. Look for deep learning–based matting networks (e.g., encoder–decoder architectures, refinement modules), novel loss functions tailored to alpha prediction, new matting datasets, or techniques that leverage trimaps, natural image priors, or auxiliary tasks (e.g., semantic segmentation) to improve matting accuracy.

Not relevant: Papers that perform general image segmentation (semantic or instance) but do not explicitly address alpha matting. Also exclude works that use matting as a subroutine in another pipeline (e.g., for portrait editing in a larger application) without proposing a novel matting algorithm or matting-specific contributions.

Video Matting

Relevant: Papers devoted to extracting alpha mattes for moving subjects in video, emphasizing temporal consistency, efficient propagation of alpha masks between frames, or the integration of motion cues (optical flow, temporal attention) into the matting network. These works often propose recurrent or 3D-CNN architectures, leverage per-frame trimaps plus propagation strategies, or introduce new benchmarks for video matting.

Not relevant: Works on video segmentation or background subtraction that do not explicitly model alpha mattes (i.e., they produce binary masks or bounding boxes, not soft alpha layers). Also exclude papers that apply image matting frame by frame without addressing temporal coherence or motion-specific challenges.

In suggesting papers based on the above topics, remember that your friend enjoys research on statistical machine learning and generative modeling in computer vision, especially methods that reveal surprising empirical findings or employ clever statistical tricks. He prefers papers proposing fundamentally new architectures or unified frameworks over those focused primarily on applications to specific datasets or domains.
