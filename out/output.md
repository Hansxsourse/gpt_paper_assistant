# Personalized Daily ArXiv Papers 06/13/2025
Total relevant papers: 3

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [VINCIE: Unlocking In-context Image Editing from Video](#link0)
**Authors:** Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, Lu Jiang

1. [Post-Training Quantization for Video Matting](#link1)
**Authors:** Tianrui Zhu, Houyuan Chen, Ruihao Gong, Michele Magno, Haotong Qin, Kai Zhang

2. [SceneCompleter: Dense 3D Scene Completion for Generative Novel View Synthesis](#link2)
**Authors:** Weiliang Chen, Jiayi Bi, Yuanhui Huang, Wenzhao Zheng, Yueqi Duan

---
## 0. [VINCIE: Unlocking In-context Image Editing from Video](https://arxiv.org/abs/2506.10941) <a id="link0"></a>
**ArXiv ID:** 2506.10941
**Authors:** Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, Lu Jiang

**Abstract:**  In-context image editing aims to modify images based on a contextual sequence comprising text and previously generated images. Existing methods typically depend on task-specific pipelines and expert models (e.g., segmentation and inpainting) to curate training data. In this work, we explore whether an in-context image editing model can be learned directly from videos. We introduce a scalable approach to annotate videos as interleaved multimodal sequences. To effectively learn from this data, we design a block-causal diffusion transformer trained on three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. Additionally, we propose a novel multi-turn image editing benchmark to advance research in this area. Extensive experiments demonstrate that our model exhibits strong in-context image editing capabilities and achieves state-of-the-art results on two multi-turn image editing benchmarks. Despite being trained exclusively on videos, our model also shows promising abilities in multi-concept composition, story generation, and chain-of-editing applications.

**Comment:** Matches criteria 1 closely as it involves a diffusion transformer for image editing and segmentation prediction.
**Relevance:** 5
**Novelty:** 7

---

## 1. [Post-Training Quantization for Video Matting](https://arxiv.org/abs/2506.10840) <a id="link1"></a>
**ArXiv ID:** 2506.10840
**Authors:** Tianrui Zhu, Houyuan Chen, Ruihao Gong, Michele Magno, Haotong Qin, Kai Zhang

**Abstract:**  Video matting is crucial for applications such as film production and virtual reality, yet deploying its computationally intensive models on resource-constrained devices presents challenges. Quantization is a key technique for model compression and acceleration. As an efficient approach, Post-Training Quantization (PTQ) is still in its nascent stages for video matting, facing significant hurdles in maintaining accuracy and temporal coherence. To address these challenges, this paper proposes a novel and general PTQ framework specifically designed for video matting models, marking, to the best of our knowledge, the first systematic attempt in this domain. Our contributions include: (1) A two-stage PTQ strategy that combines block-reconstruction-based optimization for fast, stable initial quantization and local dependency capture, followed by a global calibration of quantization parameters to minimize accuracy loss. (2) A Statistically-Driven Global Affine Calibration (GAC) method that enables the network to compensate for cumulative statistical distortions arising from factors such as neglected BN layer effects, even reducing the error of existing PTQ methods on video matting tasks up to 20%. (3) An Optical Flow Assistance (OFA) component that leverages temporal and semantic priors from frames to guide the PTQ process, enhancing the model's ability to distinguish moving foregrounds in complex scenes and ultimately achieving near full-precision performance even under ultra-low-bit quantization. Comprehensive quantitative and visual results show that our PTQ4VM achieves the state-of-the-art accuracy performance across different bit-widths compared to the existing quantization methods. We highlight that the 4-bit PTQ4VM even achieves performance close to the full-precision counterpart while enjoying 8x FLOP savings.

**Comment:** Matches criteria 4 closely as it addresses video matting with a novel PTQ framework.
**Relevance:** 5
**Novelty:** 6

---

## 2. [SceneCompleter: Dense 3D Scene Completion for Generative Novel View Synthesis](https://arxiv.org/abs/2506.10981) <a id="link2"></a>
**ArXiv ID:** 2506.10981
**Authors:** Weiliang Chen, Jiayi Bi, Yuanhui Huang, Wenzhao Zheng, Yueqi Duan

**Abstract:**  Generative models have gained significant attention in novel view synthesis (NVS) by alleviating the reliance on dense multi-view captures. However, existing methods typically fall into a conventional paradigm, where generative models first complete missing areas in 2D, followed by 3D recovery techniques to reconstruct the scene, which often results in overly smooth surfaces and distorted geometry, as generative models struggle to infer 3D structure solely from RGB data. In this paper, we propose SceneCompleter, a novel framework that achieves 3D-consistent generative novel view synthesis through dense 3D scene completion. SceneCompleter achieves both visual coherence and 3D-consistent generative scene completion through two key components: (1) a geometry-appearance dual-stream diffusion model that jointly synthesizes novel views in RGBD space; (2) a scene embedder that encodes a more holistic scene understanding from the reference image. By effectively fusing structural and textural information, our method demonstrates superior coherence and plausibility in generative novel view synthesis across diverse datasets. Project Page: https://chen-wl20.github.io/SceneCompleter

**Comment:** 
**Relevance:** 3
**Novelty:** 5

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
