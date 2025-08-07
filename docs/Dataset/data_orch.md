# Meta-Learning Data orchestration
A Meta‑Learning orchestration framework starts with covering widely used datasets and transformations, providing flexible support for custom datasets
and potentially custom data-loading and data-transfomation practices.


## Key Meta‑Learning Datasets & Benchmarks
These datasets are standard in few‑shot/meta‑learning research and often supported by meta‑learning libraries like learn2learn or Torchmeta

[A_Comprehensive_Overview_and_Survey..](https://www.researchgate.net/publication/340883874_A_Comprehensive_Overview_and_Survey_of_Recent_Advances_in_Meta-Learning)

[Few-shot_and_meta-learning_methods_for_image_understanding_a_survey](https://www.researchgate.net/publication/371953426_Few-shot_and_meta-learning_methods_for_image_understanding_a_survey)


### Vision Datasets:
- Omniglot – handwritten characters from ~1,600 classes, typically used with 5-way 1-shot tasks, augmented via 90° rotations.
</br>Ref: [The Effect of Diversity in Meta-Learning](https://ar5iv.labs.arxiv.org/html/2201.11775v2?utm_source=chatgpt.com)


- miniImageNet – 100 classes, 600 images each, resized to 84×84, split: 64 train, 16 val, 20 test
</br>Ref. [Meta-Learning for Semi-Supervised Few-Shot Classification (ar5iv)](https://ar5iv.labs.arxiv.org/html/1803.00676)


- tieredImageNet – hierarchical subset of ImageNet with 608 classes grouped by semantic categories; more challenging splits
Reddit:
</br>ar5iv: [Meta-Learning for Semi-Supervised Few-Shot Classification](https://ar5iv.labs.arxiv.org/html/1803.00676#:~:text=Meta%2DLearning%20for%20Semi%2DSupervised%20Few%2DShot%20Classification)
</br>ar5iv:
</br>ResearchGate: [Few-shot and meta-learning methods for image understanding: a survey](https://www.researchgate.net/publication/371953426_Few-shot_and_meta-learning_methods_for_image_understanding_a_survey)

- CIFAR-FS – few-shot splits derived from CIFAR‑100, used to evaluate generalization on small images (32×32)
</br>arXiv: [Meta-learning with differentiable closed-form solvers](https://www.researchgate.net/publication/325283458_Meta-learning_with_differentiable_closed-form_solvers)

- FC100 – The FC100 dataset was originally introduced by Oreshkin et al., 2018. It is based on CIFAR100, but unlike CIFAR-FS training, validation, and testing classes are
split so as to minimize the information overlap between splits.
The 100 classes are grouped into 20 superclasses of which 12 (60 classes) are used for training,
4 (20 classes) for validation, and 4 (20 classes) for testing.
Each class contains 600 images.
</br>arXiv: [TADAM: Task dependent adaptive metric for improved few-shot learning](https://arxiv.org/abs/1805.10123)
</br>paper: [Meta-Learning with Differentiable Convex Optimization (arXiv)](https://arxiv.org/abs/1904.03758) | [code](https://github.com/kjunelee/MetaOptNet)

### Meta‑Dataset & Other Meta‑Datasets
- Meta-Dataset – a dataset-of-datasets combining ImageNet, Omniglot, CUB, Aircraft, QuickDraw, Fungi, VGG Flower, Traffic Signs, MSCOCO, etc., designed for cross-domain meta‑learning
arXiv:


- Meta-Album – 40+ small, diverse real-world datasets spanning ecology, remote sensing, etc., designed for scalable few‑shot learning
</br>Reddit:

- Meta Omnium – a multi-modal meta-dataset for recognition, segmentation, keypoint localization across diverse vision tasks
</br>arXiv:

### Other Domains (Emerging)
- MetaAudio – few-shot benchmarks for audio classification across environmental and speech sound datasets
</br>arXiv: [MetaAudio: A Few-Shot Audio Classification Benchmark](https://arxiv.org/abs/2204.02121)

- Textual/general LLM datasets like BiomixQA (biomedical QA), The Pile, OSCAR, etc. can also be included when meta‑learning across NLP task formats
Reddit:

</br>
🔄 Standard Transformations & Task Generation
Meta‑Learning infrastructure typically applies these transformations:

- N‑way K‑shot task generation: sample N classes and K examples for support, plus queries.

- Class split transforms, like random splits for FC100 or tieredImageNet.

- Cross-domain sampling: Meta-Dataset or Meta-Album tasks sample from diverse datasets per episode.

- Dataset normalization/transforms: resizing, color normalization, cropping.

</br>
Libraries like [Torchmeta](https://github.com/tristandeleu/pytorch-meta) and [learn2learn](https://learn2learn.net/) provide unified APIs to wrap these
datasets, splits, and transforms.

</br>

---

### 📋 **Overview Table**

| Dataset / Benchmark       | Domain        | Key Use                                |
|---------------------------|---------------|-----------------------------------------|
| Omniglot                  | Images        | Character-level few‑shot                |
| miniImageNet / tieredImageNet | Images   | Classic few-shot benchmarks             |
| CIFAR-FS / FC100          | Images        | Small-resolution few-shot               |
| Meta-Dataset              | Multi-domain  | Cross-domain generalization             |
| Meta-Album                | Multi-domain  | Large scale diverse meta tasks          |
| Meta Omnium               | Vision tasks  | Recognition, segmentation, etc.         |
| MetaAudio                 | Audio         | Few-shot sound classification           |
| BiomixQA, Pile, OSCAR     | Text          | Meta-NLP & retrieval/meta-RAG           |


</br>

***-> Targets for this Framework***
- ✅ Use learn2learn: standardized API wrappers for meta‑learning datasets.

- ✅ Support many standard datasets (Omniglot, mini/tieredImageNet, CIFAR-FS, FC100).

- ✅ Provide Meta-Dataset, Meta-Album, Meta Omnium for cross-domain robustness.

- ✅ Include flexible transforms: N‑way, K‑shot sampling, rotations, splits.

- ✅ Allow users to plug in custom datasets via the API example above.

Example:
```[python]

from torchvision.datasets import ImageFolder
from learn2learn.data import MetaDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision import transforms

# Suppose Custom dataset folder: data/custom/{class}/images...
dataset = ImageFolder('data/custom', transform=transforms.ToTensor())

meta = MetaDataset(dataset)
transform = transforms.Compose([
    NWays(meta, n=5),
    KShots(meta, k=1 + 15),   # 1 support + K query
    LoadData(meta),
    RemapLabels(meta),
])
taskset = l2l.data.TaskDataset(meta, task_transforms=[transform], num_tasks=1000)
support, query = taskset.sample()

```
