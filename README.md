# Meta-Flame
Meta-Flame bridges the gap between reseearch, software development and production empowering the quick adaptation
of AI algorithms to Meta-Learning. Keeping productivity acrros multiple tasks in mind,
this sollution enables one model-agnostic package to boost AI applications, optimize
existing AI models and reduce development time rapidly in the AI domain.

A structured code repository for optimized runs for Meta-Learning tasks. The
main focus is Computer-Vision (CV), Generative AI and MultiModal applications.

You can utilize open-source datasets for running ML flows and connecting with API
easily, with little code knowledge.
The key is a fast and efficient AI toolbox.
Our sollution provides a user-friendly
tool for learning, optimization, research as well as enterprise sollutions.

# Values

- Economy of code
- Targeted development
- Agnosticism: one framework for all models and datasets.
- Explainable AI(xAI)/Explainable software
- Research2Development

# Introduction

Meta learning projects can quickly become complex and messy without a well-defined structure. This project template aims to address this issue by providing a consistent, organized, and extensible structure as well as an easy-to-use
template focus on fast-paced runs. With this structure, you can:

* Easily manage datasets, models, API requests, and experiments.
* Keep track of dependencies using virtual environments.
* Facilitate code collaboration among team members.
* Maintain clear documentation and logs.
* Enable reproducibility of experiments.

## Folder Structure

```
project-root/
│
├── config/
│
├── research/
│   ├── trials.ipynb
│
├── src/projName
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │   └── model_evaluation.py
│   │   └── ...
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   ├── train_pipeline.py
│   │   └── ...
│   ├── utils/
│   │   ├── logging.py
│   │   ├── config.py
│   │   └── common.py
|   |   └── ...
│   ├── constants/
|   |   └── ...
│   ├── entity/
|   |   └── ...
├── ├── config/
│   |   └── config.yaml
|
├── templates/
│   ├── index.html
|   └── ...
|
├── tests/
|   └── ...
│
├── requirements.txt
├── README.md
├── LICENSE
└── setup.py
└── params.yaml
└── main.py
└── ...

```

# Usage

## Running
The usage is split into three pipelines:

-> Data pipeline: collect, wrap the dataset and add transformation, produsing the Dataloader component
</br>Split into [Data Collect] (download/fetch) and [Data Wrap] (wrap into a set and a loader with transforms)

-> Train pipeline: Start the training loop, split into training and validation phase.

-> Predict pipeline: Production pipeline, it is the Testing Phase of the algorithm, evaluating its results in real\
data predicitons.

## Testing

Testing is split as Unit tests per component

- Data Component: test Dataloader and Dataset.
