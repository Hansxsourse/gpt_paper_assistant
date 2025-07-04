# Personalized Daily ArXiv Papers 07/04/2025
Total relevant papers: 10

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion](#link0)
**Authors:** Fangfu Liu, Hao Li, Jiawei Chi, Hanyang Wang, Minghui Yang, Fudong Wang, Yueqi Duan

1. [UniMC: Taming Diffusion Transformer for Unified Keypoint-Guided Multi-Class Image Generation](#link1)
**Authors:** Qin Guo, Ailing Zeng, Dongxu Yue, Ceyuan Yang, Yang Cao, Hanzhong Guo, Fei Shen, Wei Liu, Xihui Liu, Dan Xu

2. [USAD: An Unsupervised Data Augmentation Spatio-Temporal Attention Diffusion Network](#link2)
**Authors:** Ying Yu, Hang Xiao, Siyao Li, Jiarui Li, Haotian Tang, Hanyu Liu, Chao Li

3. [PosDiffAE: Position-aware Diffusion Auto-encoder For High-Resolution Brain Tissue Classification Incorporating Artifact Restoration](#link3)
**Authors:** Ayantika Das, Moitreya Chaudhuri, Koushik Bhat, Keerthi Ram, Mihail Bota, Mohanasankar Sivaprakasam

4. [SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment](#link4)
**Authors:** Qi Xu, Dongxu Wei, Lingzhe Zhao, Wenpu Li, Zhangchi Huang, Shunping Ji, Peidong Liu

5. [AnyI2V: Animating Any Conditional Image with Motion Control](#link5)
**Authors:** Ziye Li, Hao Luo, Xincheng Shuai, Henghui Ding

6. [RefTok: Reference-Based Tokenization for Video Generation](#link6)
**Authors:** Xiang Fan, Xiaohang Sun, Kushan Thakkar, Zhu Liu, Vimal Bhat, Ranjay Krishna, Xiang Hao

7. [DreamComposer++: Empowering Diffusion Models with Multi-View Conditions for 3D Content Generation](#link7)
**Authors:** Yunhan Yang, Shuo Chen, Yukun Huang, Xiaoyang Wu, Yuan-Chen Guo, Edmund Y. Lam, Hengshuang Zhao, Tong He, Xihui Liu

8. [Heeding the Inner Voice: Aligning ControlNet Training via Intermediate Features Feedback](#link8)
**Authors:** Nina Konovalova, Maxim Nikolaev, Andrey Kuznetsov, Aibek Alanov

9. [High-Fidelity Differential-information Driven Binary Vision Transformer](#link9)
**Authors:** Tian Gao, Zhiyuan Zhang, Kaijie Yin, Xu-Cheng Zhong, Hui Kong

---
## 0. [LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion](https://arxiv.org/abs/2507.02813) <a id="link0"></a>
**ArXiv ID:** 2507.02813
**Authors:** Fangfu Liu, Hao Li, Jiawei Chi, Hanyang Wang, Minghui Yang, Fudong Wang, Yueqi Duan

**Abstract:**  Recovering 3D structures with open-vocabulary scene understanding from 2D images is a fundamental but daunting task. Recent developments have achieved this by performing per-scene optimization with embedded language information. However, they heavily rely on the calibrated dense-view reconstruction paradigm, thereby suffering from severe rendering artifacts and implausible semantic synthesis when limited views are available. In this paper, we introduce a novel generative framework, coined LangScene-X, to unify and generate 3D consistent multi-modality information for reconstruction and understanding. Powered by the generative capability of creating more consistent novel observations, we can build generalizable 3D language-embedded scenes from only sparse views. Specifically, we first train a TriMap video diffusion model that can generate appearance (RGBs), geometry (normals), and semantics (segmentation maps) from sparse inputs through progressive knowledge integration. Furthermore, we propose a Language Quantized Compressor (LQC), trained on large-scale image datasets, to efficiently encode language embeddings, enabling cross-scene generalization without per-scene retraining. Finally, we reconstruct the language surface fields by aligning language information onto the surface of 3D scenes, enabling open-ended language queries. Extensive experiments on real-world data demonstrate the superiority of our LangScene-X over state-of-the-art methods in terms of quality and generalizability. Project Page: https://liuff19.github.io/LangScene-X.

**Comment:** Matches criteria 1 and 2 closely
**Relevance:** 10
**Novelty:** 8

---

## 1. [UniMC: Taming Diffusion Transformer for Unified Keypoint-Guided Multi-Class Image Generation](https://arxiv.org/abs/2507.02713) <a id="link1"></a>
**ArXiv ID:** 2507.02713
**Authors:** Qin Guo, Ailing Zeng, Dongxu Yue, Ceyuan Yang, Yang Cao, Hanzhong Guo, Fei Shen, Wei Liu, Xihui Liu, Dan Xu

**Abstract:**  Although significant advancements have been achieved in the progress of keypoint-guided Text-to-Image diffusion models, existing mainstream keypoint-guided models encounter challenges in controlling the generation of more general non-rigid objects beyond humans (e.g., animals). Moreover, it is difficult to generate multiple overlapping humans and animals based on keypoint controls solely. These challenges arise from two main aspects: the inherent limitations of existing controllable methods and the lack of suitable datasets. First, we design a DiT-based framework, named UniMC, to explore unifying controllable multi-class image generation. UniMC integrates instance- and keypoint-level conditions into compact tokens, incorporating attributes such as class, bounding box, and keypoint coordinates. This approach overcomes the limitations of previous methods that struggled to distinguish instances and classes due to their reliance on skeleton images as conditions. Second, we propose HAIG-2.9M, a large-scale, high-quality, and diverse dataset designed for keypoint-guided human and animal image generation. HAIG-2.9M includes 786K images with 2.9M instances. This dataset features extensive annotations such as keypoints, bounding boxes, and fine-grained captions for both humans and animals, along with rigorous manual inspection to ensure annotation accuracy. Extensive experiments demonstrate the high quality of HAIG-2.9M and the effectiveness of UniMC, particularly in heavy occlusions and multi-class scenarios.

**Comment:** 2
**Relevance:** 5
**Novelty:** 6

---

## 2. [USAD: An Unsupervised Data Augmentation Spatio-Temporal Attention Diffusion Network](https://arxiv.org/abs/2507.02827) <a id="link2"></a>
**ArXiv ID:** 2507.02827
**Authors:** Ying Yu, Hang Xiao, Siyao Li, Jiarui Li, Haotian Tang, Hanyu Liu, Chao Li

**Abstract:**  The primary objective of human activity recognition (HAR) is to infer ongoing human actions from sensor data, a task that finds broad applications in health monitoring, safety protection, and sports analysis. Despite proliferating research, HAR still faces key challenges, including the scarcity of labeled samples for rare activities, insufficient extraction of high-level features, and suboptimal model performance on lightweight devices. To address these issues, this paper proposes a comprehensive optimization approach centered on multi-attention interaction mechanisms. First, an unsupervised, statistics-guided diffusion model is employed to perform data augmentation, thereby alleviating the problems of labeled data scarcity and severe class imbalance. Second, a multi-branch spatio-temporal interaction network is designed, which captures multi-scale features of sequential data through parallel residual branches with 3*3, 5*5, and 7*7 convolutional kernels. Simultaneously, temporal attention mechanisms are incorporated to identify critical time points, while spatial attention enhances inter-sensor interactions. A cross-branch feature fusion unit is further introduced to improve the overall feature representation capability. Finally, an adaptive multi-loss function fusion strategy is integrated, allowing for dynamic adjustment of loss weights and overall model optimization. Experimental results on three public datasets, WISDM, PAMAP2, and OPPORTUNITY, demonstrate that the proposed unsupervised data augmentation spatio-temporal attention diffusion network (USAD) achieves accuracies of 98.84%, 93.81%, and 80.92% respectively, significantly outperforming existing approaches. Furthermore, practical deployment on embedded devices verifies the efficiency and feasibility of the proposed method.

**Comment:** 2
**Relevance:** 5
**Novelty:** 6

---

## 3. [PosDiffAE: Position-aware Diffusion Auto-encoder For High-Resolution Brain Tissue Classification Incorporating Artifact Restoration](https://arxiv.org/abs/2507.02405) <a id="link3"></a>
**ArXiv ID:** 2507.02405
**Authors:** Ayantika Das, Moitreya Chaudhuri, Koushik Bhat, Keerthi Ram, Mihail Bota, Mohanasankar Sivaprakasam

**Abstract:**  Denoising diffusion models produce high-fidelity image samples by capturing the image distribution in a progressive manner while initializing with a simple distribution and compounding the distribution complexity. Although these models have unlocked new applicabilities, the sampling mechanism of diffusion does not offer means to extract image-specific semantic representation, which is inherently provided by auto-encoders. The encoding component of auto-encoders enables mapping between a specific image and its latent space, thereby offering explicit means of enforcing structures in the latent space. By integrating an encoder with the diffusion model, we establish an auto-encoding formulation, which learns image-specific representations and offers means to organize the latent space. In this work, First, we devise a mechanism to structure the latent space of a diffusion auto-encoding model, towards recognizing region-specific cellular patterns in brain images. We enforce the representations to regress positional information of the patches from high-resolution images. This creates a conducive latent space for differentiating tissue types of the brain. Second, we devise an unsupervised tear artifact restoration technique based on neighborhood awareness, utilizing latent representations and the constrained generation capability of diffusion models during inference. Third, through representational guidance and leveraging the inference time steerable noising and denoising capability of diffusion, we devise an unsupervised JPEG artifact restoration technique.

**Comment:** Does not match any specific criteria
**Relevance:** 3
**Novelty:** 6

---

## 4. [SIU3R: Simultaneous Scene Understanding and 3D Reconstruction Beyond Feature Alignment](https://arxiv.org/abs/2507.02705) <a id="link4"></a>
**ArXiv ID:** 2507.02705
**Authors:** Qi Xu, Dongxu Wei, Lingzhe Zhao, Wenpu Li, Zhangchi Huang, Shunping Ji, Peidong Liu

**Abstract:**  Simultaneous understanding and 3D reconstruction plays an important role in developing end-to-end embodied intelligent systems. To achieve this, recent approaches resort to 2D-to-3D feature alignment paradigm, which leads to limited 3D understanding capability and potential semantic information loss. In light of this, we propose SIU3R, the first alignment-free framework for generalizable simultaneous understanding and 3D reconstruction from unposed images. Specifically, SIU3R bridges reconstruction and understanding tasks via pixel-aligned 3D representation, and unifies multiple understanding tasks into a set of unified learnable queries, enabling native 3D understanding without the need of alignment with 2D models. To encourage collaboration between the two tasks with shared representation, we further conduct in-depth analyses of their mutual benefits, and propose two lightweight modules to facilitate their interaction. Extensive experiments demonstrate that our method achieves state-of-the-art performance not only on the individual tasks of 3D reconstruction and understanding, but also on the task of simultaneous understanding and 3D reconstruction, highlighting the advantages of our alignment-free framework and the effectiveness of the mutual benefit designs.

**Comment:** 
**Relevance:** 3
**Novelty:** 6

---

## 5. [AnyI2V: Animating Any Conditional Image with Motion Control](https://arxiv.org/abs/2507.02857) <a id="link5"></a>
**ArXiv ID:** 2507.02857
**Authors:** Ziye Li, Hao Luo, Xincheng Shuai, Henghui Ding

**Abstract:**  Recent advancements in video generation, particularly in diffusion models, have driven notable progress in text-to-video (T2V) and image-to-video (I2V) synthesis. However, challenges remain in effectively integrating dynamic motion signals and flexible spatial constraints. Existing T2V methods typically rely on text prompts, which inherently lack precise control over the spatial layout of generated content. In contrast, I2V methods are limited by their dependence on real images, which restricts the editability of the synthesized content. Although some methods incorporate ControlNet to introduce image-based conditioning, they often lack explicit motion control and require computationally expensive training. To address these limitations, we propose AnyI2V, a training-free framework that animates any conditional images with user-defined motion trajectories. AnyI2V supports a broader range of modalities as the conditional image, including data types such as meshes and point clouds that are not supported by ControlNet, enabling more flexible and versatile video generation. Additionally, it supports mixed conditional inputs and enables style transfer and editing via LoRA and text prompts. Extensive experiments demonstrate that the proposed AnyI2V achieves superior performance and provides a new perspective in spatial- and motion-controlled video generation. Code is available at https://henghuiding.com/AnyI2V/.

**Comment:** 
**Relevance:** 3
**Novelty:** 5

---

## 6. [RefTok: Reference-Based Tokenization for Video Generation](https://arxiv.org/abs/2507.02862) <a id="link6"></a>
**ArXiv ID:** 2507.02862
**Authors:** Xiang Fan, Xiaohang Sun, Kushan Thakkar, Zhu Liu, Vimal Bhat, Ranjay Krishna, Xiang Hao

**Abstract:**  Effectively handling temporal redundancy remains a key challenge in learning video models. Prevailing approaches often treat each set of frames independently, failing to effectively capture the temporal dependencies and redundancies inherent in videos. To address this limitation, we introduce RefTok, a novel reference-based tokenization method capable of capturing complex temporal dynamics and contextual information. Our method encodes and decodes sets of frames conditioned on an unquantized reference frame. When decoded, RefTok preserves the continuity of motion and the appearance of objects across frames. For example, RefTok retains facial details despite head motion, reconstructs text correctly, preserves small patterns, and maintains the legibility of handwriting from the context. Across 4 video datasets (K600, UCF-101, BAIR Robot Pushing, and DAVIS), RefTok significantly outperforms current state-of-the-art tokenizers (Cosmos and MAGVIT) and improves all evaluated metrics (PSNR, SSIM, LPIPS) by an average of 36.7% at the same or higher compression ratios. When a video generation model is trained using RefTok's latents on the BAIR Robot Pushing task, the generations not only outperform MAGVIT-B but the larger MAGVIT-L, which has 4x more parameters, across all generation metrics by an average of 27.9%.

**Comment:** 
**Relevance:** 3
**Novelty:** 5

---

## 7. [DreamComposer++: Empowering Diffusion Models with Multi-View Conditions for 3D Content Generation](https://arxiv.org/abs/2507.02299) <a id="link7"></a>
**ArXiv ID:** 2507.02299
**Authors:** Yunhan Yang, Shuo Chen, Yukun Huang, Xiaoyang Wu, Yuan-Chen Guo, Edmund Y. Lam, Hengshuang Zhao, Tong He, Xihui Liu

**Abstract:**  Recent advancements in leveraging pre-trained 2D diffusion models achieve the generation of high-quality novel views from a single in-the-wild image. However, existing works face challenges in producing controllable novel views due to the lack of information from multiple views. In this paper, we present DreamComposer++, a flexible and scalable framework designed to improve current view-aware diffusion models by incorporating multi-view conditions. Specifically, DreamComposer++ utilizes a view-aware 3D lifting module to extract 3D representations of an object from various views. These representations are then aggregated and rendered into the latent features of target view through the multi-view feature fusion module. Finally, the obtained features of target view are integrated into pre-trained image or video diffusion models for novel view synthesis. Experimental results demonstrate that DreamComposer++ seamlessly integrates with cutting-edge view-aware diffusion models and enhances their abilities to generate controllable novel views from multi-view conditions. This advancement facilitates controllable 3D object reconstruction and enables a wide range of applications.

**Comment:** 
**Relevance:** 3
**Novelty:** 5

---

## 8. [Heeding the Inner Voice: Aligning ControlNet Training via Intermediate Features Feedback](https://arxiv.org/abs/2507.02321) <a id="link8"></a>
**ArXiv ID:** 2507.02321
**Authors:** Nina Konovalova, Maxim Nikolaev, Andrey Kuznetsov, Aibek Alanov

**Abstract:**  Despite significant progress in text-to-image diffusion models, achieving precise spatial control over generated outputs remains challenging. ControlNet addresses this by introducing an auxiliary conditioning module, while ControlNet++ further refines alignment through a cycle consistency loss applied only to the final denoising steps. However, this approach neglects intermediate generation stages, limiting its effectiveness. We propose InnerControl, a training strategy that enforces spatial consistency across all diffusion steps. Our method trains lightweight convolutional probes to reconstruct input control signals (e.g., edges, depth) from intermediate UNet features at every denoising step. These probes efficiently extract signals even from highly noisy latents, enabling pseudo ground truth controls for training. By minimizing the discrepancy between predicted and target conditions throughout the entire diffusion process, our alignment loss improves both control fidelity and generation quality. Combined with established techniques like ControlNet++, InnerControl achieves state-of-the-art performance across diverse conditioning methods (e.g., edges, depth).

**Comment:** 
**Relevance:** 3
**Novelty:** 5

---

## 9. [High-Fidelity Differential-information Driven Binary Vision Transformer](https://arxiv.org/abs/2507.02222) <a id="link9"></a>
**ArXiv ID:** 2507.02222
**Authors:** Tian Gao, Zhiyuan Zhang, Kaijie Yin, Xu-Cheng Zhong, Hui Kong

**Abstract:**  The binarization of vision transformers (ViTs) offers a promising approach to addressing the trade-off between high computational/storage demands and the constraints of edge-device deployment. However, existing binary ViT methods often suffer from severe performance degradation or rely heavily on full-precision modules. To address these issues, we propose DIDB-ViT, a novel binary ViT that is highly informative while maintaining the original ViT architecture and computational efficiency. Specifically, we design an informative attention module incorporating differential information to mitigate information loss caused by binarization and enhance high-frequency retention. To preserve the fidelity of the similarity calculations between binary Q and K tensors, we apply frequency decomposition using the discrete Haar wavelet and integrate similarities across different frequencies. Additionally, we introduce an improved RPReLU activation function to restructure the activation distribution, expanding the model's representational capacity. Experimental results demonstrate that our DIDB-ViT significantly outperforms state-of-the-art network quantization methods in multiple ViT architectures, achieving superior image classification and segmentation performance.

**Comment:** 
**Relevance:** 3
**Novelty:** 4

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
