# Personalized Daily ArXiv Papers 06/05/2025
Total relevant papers: 11

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [FullDiT2: Efficient In-Context Conditioning for Video Diffusion Transformers](#link0)
**Authors:** Xuanhua He, Quande Liu, Zixuan Ye, Wecai Ye, Qiulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan, Di Zhang, Kun Gai

1. [Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation](#link1)
**Authors:** Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson W. H. Lau, Wangmeng Zuo, Chunchao Guo

2. [LayerFlow: A Unified Model for Layer-aware Video Generation](#link2)
**Authors:** Sihui Ji, Hao Luo, Xi Chen, Yuanpeng Tu, Yiyang Wang, Hengshuang Zhao

3. [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](#link3)
**Authors:** Ziyi Wu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Ashkan Mirzaei, Igor Gilitschenski, Sergey Tulyakov, Aliaksandr Siarohin

4. [Resolving Task Objective Conflicts in Unified Multimodal Understanding and Generation via Task-Aware Mixture-of-Experts](#link4)
**Authors:** Jiaxing Zhang, Xinyi Zeng, Hao Tang

5. [JointSplat: Probabilistic Joint Flow-Depth Optimization for Sparse-View Gaussian Splatting](#link5)
**Authors:** Yang Xiao, Guoan Xu, Qiang Wu, Wenjing Jia

6. [UNIC: Unified In-Context Video Editing](#link6)
**Authors:** Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, Wenhan Luo

7. [ControlThinker: Unveiling Latent Semantics for Controllable Image Generation through Visual Reasoning](#link7)
**Authors:** Feng Han, Yang Jiao, Shaoxiang Chen, Junhao Xu, Jingjing Chen, Yu-Gang Jiang

8. [A Large-Scale Referring Remote Sensing Image Segmentation Dataset and Benchmark](#link8)
**Authors:** Zhigang Yang, Huiguang Yao, Linmao Tian, Xuezhi Zhao, Qiang Li, Qi Wang

9. [UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation](#link9)
**Authors:** Jinting Wang, Shan Yang, Li Liu

10. [Channel-adaptive Cross-modal Generative Semantic Communication for Point Cloud Transmission](#link10)
**Authors:** Wanting Yang, Zehui Xiong, Qianqian Yang, Ping Zhang, Merouane Debbah, Rahim Tafazolli

---
## 0. [FullDiT2: Efficient In-Context Conditioning for Video Diffusion Transformers](https://arxiv.org/abs/2506.04213) <a id="link0"></a>
**ArXiv ID:** 2506.04213
**Authors:** Xuanhua He, Quande Liu, Zixuan Ye, Wecai Ye, Qiulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan, Di Zhang, Kun Gai

**Abstract:**  Fine-grained and efficient controllability on video diffusion transformers has raised increasing desires for the applicability. Recently, In-context Conditioning emerged as a powerful paradigm for unified conditional video generation, which enables diverse controls by concatenating varying context conditioning signals with noisy video latents into a long unified token sequence and jointly processing them via full-attention, e.g., FullDiT. Despite their effectiveness, these methods face quadratic computation overhead as task complexity increases, hindering practical deployment. In this paper, we study the efficiency bottleneck neglected in original in-context conditioning video generation framework. We begin with systematic analysis to identify two key sources of the computation inefficiencies: the inherent redundancy within context condition tokens and the computational redundancy in context-latent interactions throughout the diffusion process. Based on these insights, we propose FullDiT2, an efficient in-context conditioning framework for general controllability in both video generation and editing tasks, which innovates from two key perspectives. Firstly, to address the token redundancy, FullDiT2 leverages a dynamic token selection mechanism to adaptively identify important context tokens, reducing the sequence length for unified full-attention. Additionally, a selective context caching mechanism is devised to minimize redundant interactions between condition tokens and video latents. Extensive experiments on six diverse conditional video editing and generation tasks demonstrate that FullDiT2 achieves significant computation reduction and 2-3 times speedup in averaged time cost per diffusion step, with minimal degradation or even higher performance in video generation quality. The project page is at \href{https://fulldit2.github.io/}{https://fulldit2.github.io/}.

**Comment:** Matches criterion 2: Unified Diffusion Models for video generation and editing tasks.
**Relevance:** 5
**Novelty:** 7

---

## 1. [Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation](https://arxiv.org/abs/2506.04225) <a id="link1"></a>
**ArXiv ID:** 2506.04225
**Authors:** Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson W. H. Lau, Wangmeng Zuo, Chunchao Guo

**Abstract:**  Real-world applications like video gaming and virtual reality often demand the ability to model 3D scenes that users can explore along custom camera trajectories. While significant progress has been made in generating 3D objects from text or images, creating long-range, 3D-consistent, explorable 3D scenes remains a complex and challenging problem. In this work, we present Voyager, a novel video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image with user-defined camera path. Unlike existing approaches, Voyager achieves end-to-end scene generation and reconstruction with inherent consistency across frames, eliminating the need for 3D reconstruction pipelines (e.g., structure-from-motion or multi-view stereo). Our method integrates three key components: 1) World-Consistent Video Diffusion: A unified architecture that jointly generates aligned RGB and depth video sequences, conditioned on existing world observation to ensure global coherence 2) Long-Range World Exploration: An efficient world cache with point culling and an auto-regressive inference with smooth video sampling for iterative scene extension with context-aware consistency, and 3) Scalable Data Engine: A video reconstruction pipeline that automates camera pose estimation and metric depth prediction for arbitrary videos, enabling large-scale, diverse training data curation without manual 3D annotations. Collectively, these designs result in a clear improvement over existing methods in visual quality and geometric accuracy, with versatile applications.

**Comment:** Matches criterion 2: Unified Diffusion Models for multiple vision tasks including image generation and depth estimation.
**Relevance:** 5
**Novelty:** 7

---

## 2. [LayerFlow: A Unified Model for Layer-aware Video Generation](https://arxiv.org/abs/2506.04228) <a id="link2"></a>
**ArXiv ID:** 2506.04228
**Authors:** Sihui Ji, Hao Luo, Xi Chen, Yuanpeng Tu, Yiyang Wang, Hengshuang Zhao

**Abstract:**  We present LayerFlow, a unified solution for layer-aware video generation. Given per-layer prompts, LayerFlow generates videos for the transparent foreground, clean background, and blended scene. It also supports versatile variants like decomposing a blended video or generating the background for the given foreground and vice versa. Starting from a text-to-video diffusion transformer, we organize the videos for different layers as sub-clips, and leverage layer embeddings to distinguish each clip and the corresponding layer-wise prompts. In this way, we seamlessly support the aforementioned variants in one unified framework. For the lack of high-quality layer-wise training videos, we design a multi-stage training strategy to accommodate static images with high-quality layer annotations. Specifically, we first train the model with low-quality video data. Then, we tune a motion LoRA to make the model compatible with static frames. Afterward, we train the content LoRA on the mixture of image data with high-quality layered images along with copy-pasted video data. During inference, we remove the motion LoRA thus generating smooth videos with desired layers.

**Comment:** Matches criteria 1 closely: unified framework for video generation with layer-aware segmentation.
**Relevance:** 5
**Novelty:** 6

---

## 3. [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](https://arxiv.org/abs/2506.03517) <a id="link3"></a>
**ArXiv ID:** 2506.03517
**Authors:** Ziyi Wu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Ashkan Mirzaei, Igor Gilitschenski, Sergey Tulyakov, Aliaksandr Siarohin

**Abstract:**  Direct Preference Optimization (DPO) has recently been applied as a post-training technique for text-to-video diffusion models. To obtain training data, annotators are asked to provide preferences between two videos generated from independent noise. However, this approach prohibits fine-grained comparisons, and we point out that it biases the annotators towards low-motion clips as they often contain fewer visual artifacts. In this work, we introduce DenseDPO, a method that addresses these shortcomings by making three contributions. First, we create each video pair for DPO by denoising corrupted copies of a ground truth video. This results in aligned pairs with similar motion structures while differing in local details, effectively neutralizing the motion bias. Second, we leverage the resulting temporal alignment to label preferences on short segments rather than entire clips, yielding a denser and more precise learning signal. With only one-third of the labeled data, DenseDPO greatly improves motion generation over vanilla DPO, while matching it in text alignment, visual quality, and temporal consistency. Finally, we show that DenseDPO unlocks automatic preference annotation using off-the-shelf Vision Language Models (VLMs): GPT accurately predicts segment-level preferences similar to task-specifically fine-tuned video reward models, and DenseDPO trained on these labels achieves performance close to using human labels.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 6

---

## 4. [Resolving Task Objective Conflicts in Unified Multimodal Understanding and Generation via Task-Aware Mixture-of-Experts](https://arxiv.org/abs/2506.03591) <a id="link4"></a>
**ArXiv ID:** 2506.03591
**Authors:** Jiaxing Zhang, Xinyi Zeng, Hao Tang

**Abstract:**  Unified multimodal large language models (MLLMs) based on end-to-end autoregressive (AR) transformers effectively integrate both understanding and generation tasks within a single framework. However, intrinsic Task Objective Conflicts between high-level semantic abstraction in understanding and fine-grained detail preservation in generation pose significant challenges, often leading to suboptimal trade-offs and task interference. Existing solutions, such as decoupling shared visual encoders, fall short of fundamentally resolving these conflicts due to inherent AR architecture. In this paper, we propose a novel approach that decouples internal components of AR to resolve task objective conflicts. Specifically, we design UTAMoE, a Unified Task-Aware Mixture-of-Experts (MoE) framework that decouples internal AR modules via a Task-Aware MoE Layer to create task-specific optimization subpaths. To enhance task differentiation while maintaining overall coordination, we introduce a novel Two-Stage Training Strategy. Extensive experiments on multimodal benchmarks demonstrate that UTAMoE mitigates task objective conflicts, achieving state-of-the-art performance across various tasks. Visualizations and ablation studies further validate the effectiveness of our approach.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 6

---

## 5. [JointSplat: Probabilistic Joint Flow-Depth Optimization for Sparse-View Gaussian Splatting](https://arxiv.org/abs/2506.03872) <a id="link5"></a>
**ArXiv ID:** 2506.03872
**Authors:** Yang Xiao, Guoan Xu, Qiang Wu, Wenjing Jia

**Abstract:**  Reconstructing 3D scenes from sparse viewpoints is a long-standing challenge with wide applications. Recent advances in feed-forward 3D Gaussian sparse-view reconstruction methods provide an efficient solution for real-time novel view synthesis by leveraging geometric priors learned from large-scale multi-view datasets and computing 3D Gaussian centers via back-projection. Despite offering strong geometric cues, both feed-forward multi-view depth estimation and flow-depth joint estimation face key limitations: the former suffers from mislocation and artifact issues in low-texture or repetitive regions, while the latter is prone to local noise and global inconsistency due to unreliable matches when ground-truth flow supervision is unavailable. To overcome this, we propose JointSplat, a unified framework that leverages the complementarity between optical flow and depth via a novel probabilistic optimization mechanism. Specifically, this pixel-level mechanism scales the information fusion between depth and flow based on the matching probability of optical flow during training. Building upon the above mechanism, we further propose a novel multi-view depth-consistency loss to leverage the reliability of supervision while suppressing misleading gradients in uncertain areas. Evaluated on RealEstate10K and ACID, JointSplat consistently outperforms state-of-the-art (SOTA) methods, demonstrating the effectiveness and robustness of our proposed probabilistic joint flow-depth optimization approach for high-fidelity sparse-view 3D reconstruction.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 6

---

## 6. [UNIC: Unified In-Context Video Editing](https://arxiv.org/abs/2506.04216) <a id="link6"></a>
**ArXiv ID:** 2506.04216
**Authors:** Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, Wenhan Luo

**Abstract:**  Recent advances in text-to-video generation have sparked interest in generative video editing tasks. Previous methods often rely on task-specific architectures (e.g., additional adapter modules) or dedicated customizations (e.g., DDIM inversion), which limit the integration of versatile editing conditions and the unification of various editing tasks. In this paper, we introduce UNified In-Context Video Editing (UNIC), a simple yet effective framework that unifies diverse video editing tasks within a single model in an in-context manner. To achieve this unification, we represent the inputs of various video editing tasks as three types of tokens: the source video tokens, the noisy video latent, and the multi-modal conditioning tokens that vary according to the specific editing task. Based on this formulation, our key insight is to integrate these three types into a single consecutive token sequence and jointly model them using the native attention operations of DiT, thereby eliminating the need for task-specific adapter designs. Nevertheless, direct task unification under this framework is challenging, leading to severe token collisions and task confusion due to the varying video lengths and diverse condition modalities across tasks. To address these, we introduce task-aware RoPE to facilitate consistent temporal positional encoding, and condition bias that enables the model to clearly differentiate different editing tasks. This allows our approach to adaptively perform different video editing tasks by referring the source video and varying condition tokens "in context", and support flexible task composition. To validate our method, we construct a unified video editing benchmark containing six representative video editing tasks. Results demonstrate that our unified approach achieves superior performance on each task and exhibits emergent task composition abilities.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 6

---

## 7. [ControlThinker: Unveiling Latent Semantics for Controllable Image Generation through Visual Reasoning](https://arxiv.org/abs/2506.03596) <a id="link7"></a>
**ArXiv ID:** 2506.03596
**Authors:** Feng Han, Yang Jiao, Shaoxiang Chen, Junhao Xu, Jingjing Chen, Yu-Gang Jiang

**Abstract:**  The field of controllable image generation has seen significant advancements, with various architectures improving generation layout consistency with control signals. However, contemporary methods still face challenges in bridging the semantic gap between input text prompts with sparse semantics and the target images, often over-relying on low-level control signals to infer regional details. To address this challenge, we propose ControlThinker, a novel framework that employs a "comprehend-then-generate" paradigm. Firstly, by incentivizing the visual reasoning capability of a MLLM, latent semantics from control images are mined to enrich text prompts. This enriched semantic understanding then seamlessly aids in image generation without the need for additional complex modifications. To further tackle the uncertainty arising from the ambiguity of control images, we encourage broader exploration of reasoning trajectories and select the optimal one using a metric-based output reward model (ORM). Extensive experimental results demonstrate that ControlThinker effectively mitigates the semantic gap between raw text prompts and target images, resulting in improved visual quality and semantic consistency across a wide range of benchmarks. The code and models are available at https://github.com/Maplebb/ControlThinker.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 6

---

## 8. [A Large-Scale Referring Remote Sensing Image Segmentation Dataset and Benchmark](https://arxiv.org/abs/2506.03583) <a id="link8"></a>
**ArXiv ID:** 2506.03583
**Authors:** Zhigang Yang, Huiguang Yao, Linmao Tian, Xuezhi Zhao, Qiang Li, Qi Wang

**Abstract:**  Referring Remote Sensing Image Segmentation is a complex and challenging task that integrates the paradigms of computer vision and natural language processing. Existing datasets for RRSIS suffer from critical limitations in resolution, scene diversity, and category coverage, which hinders the generalization and real-world applicability of refer segmentation models. To facilitate the development of this field, we introduce NWPU-Refer, the largest and most diverse RRSIS dataset to date, comprising 15,003 high-resolution images (1024-2048px) spanning 30+ countries with 49,745 annotated targets supporting single-object, multi-object, and non-object segmentation scenarios. Additionally, we propose the Multi-scale Referring Segmentation Network (MRSNet), a novel framework tailored for the unique demands of RRSIS. MRSNet introduces two key innovations: (1) an Intra-scale Feature Interaction Module (IFIM) that captures fine-grained details within each encoder stage, and (2) a Hierarchical Feature Interaction Module (HFIM) to enable seamless cross-scale feature fusion, preserving spatial integrity while enhancing discriminative power. Extensive experiments conducte on the proposed NWPU-Refer dataset demonstrate that MRSNet achieves state-of-the-art performance across multiple evaluation metrics, validating its effectiveness. The dataset and code are publicly available at https://github.com/CVer-Yang/NWPU-Refer.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 5

---

## 9. [UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation](https://arxiv.org/abs/2506.04134) <a id="link9"></a>
**ArXiv ID:** 2506.04134
**Authors:** Jinting Wang, Shan Yang, Li Liu

**Abstract:**  Cued Speech (CS) enhances lipreading through hand coding, providing precise speech perception support for the hearing-impaired. CS Video-to-Speech generation (CSV2S) task aims to convert the CS visual expressions (CS videos) of hearing-impaired individuals into comprehensible speech signals. Direct generation of speech from CS video (called single CSV2S) yields poor performance due to insufficient CS data. Current research mostly focuses on CS Recognition (CSR), which convert video content into linguistic text. Based on this, one straightforward way of CSV2S is to combine CSR with a Text-to-Speech system. This combined architecture relies on text as an intermediate medium for stepwise cross-modal alignment, which may lead to error propagation and temporal misalignment between speech and video dynamics. To address these challenges, we propose a novel approach that directly generates speech from CS videos without relying on intermediate text. Building upon this, we propose UniCUE, the first unified framework for CSV2S, whose core innovation lies in the integration of the CSR task that provides fine-grained visual-semantic information to facilitate speech generation from CS videos. More precisely, (1) a novel fine-grained semantic alignment pool to ensure precise mapping between visual features and speech contents; (2) a VisioPhonetic adapter to bridge cross-task representations, ensuring seamless compatibility between two distinct tasks (i.e., CSV2S and CSR); (3) a pose-aware visual processor is introduced to enhance fine-grained spatiotemporal correlations between lip and hand movements in CS video. Experiments on our new established Chinese CS dataset (14 cuers1: 8 hearing-impaired and 6 normal-hearing) show that our UniCUE significantly reduces Word Error Rate by 78.3% and improves lip-speech synchronization by 32% compared to the single CSV2S.

**Comment:** Does not match any specific criteria.
**Relevance:** 3
**Novelty:** 5

---

## 10. [Channel-adaptive Cross-modal Generative Semantic Communication for Point Cloud Transmission](https://arxiv.org/abs/2506.03211) <a id="link10"></a>
**ArXiv ID:** 2506.03211
**Authors:** Wanting Yang, Zehui Xiong, Qianqian Yang, Ping Zhang, Merouane Debbah, Rahim Tafazolli

**Abstract:**  With the rapid development of autonomous driving and extended reality, efficient transmission of point clouds (PCs) has become increasingly important. In this context, we propose a novel channel-adaptive cross-modal generative semantic communication (SemCom) for PC transmission, called GenSeC-PC. GenSeC-PC employs a semantic encoder that fuses images and point clouds, where images serve as non-transmitted side information. Meanwhile, the decoder is built upon the backbone of PointDif. Such a cross-modal design not only ensures high compression efficiency but also delivers superior reconstruction performance compared to PointDif. Moreover, to ensure robust transmission and reduce system complexity, we design a streamlined and asymmetric channel-adaptive joint semantic-channel coding architecture, where only the encoder needs the feedback of average signal-to-noise ratio (SNR) and available bandwidth. In addition, rectified denoising diffusion implicit models is employed to accelerate the decoding process to the millisecond level, enabling real-time PC communication. Unlike existing methods, GenSeC-PC leverages generative priors to ensure reliable reconstruction even from noisy or incomplete source PCs. More importantly, it supports fully analog transmission, improving compression efficiency by eliminating the need for error-free side information transmission common in prior SemCom approaches. Simulation results confirm the effectiveness of cross-modal semantic extraction and dual-metric guided fine-tuning, highlighting the framework's robustness across diverse conditions, including low SNR, bandwidth limitations, varying numbers of 2D images, and previously unseen objects.

**Comment:** No criteria match closely.
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
