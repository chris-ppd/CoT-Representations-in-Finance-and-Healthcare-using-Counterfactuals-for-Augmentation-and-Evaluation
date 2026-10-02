Cost-Efficient LLM Distillation for Finance and Healthcare using Counterfactuals
MSc Thesis — Artificial Intelligence, Maastricht University

A framework that transfers the reasoning of a large language model to lightweightBERT-based classifiers, so predictions run without ever calling the LLM at inference time.

The Problem
Large language models are powerful for high-stakes prediction tasks like loan defaultand patient disposition, but they are expensive to run, hard to deploy, and difficultto explain and monitor. Lightweight classifiers are cheap and explainable, but lackthe reasoning depth.

The Approach
The framework uses the LLM only once, during training data preparation:

A teacher LLM (Qwen3-4B, 8-bit quantized) generates structured Chain-of-Thoughtreasoning profiles for each data point, guided by a constrained system prompt
Generated reasoning passes automatic structural and faithfulness validationbefore use
BERT-based student classifiers (~110M parameters) are trained via tworeasoning-transfer methods:
Explicit: training directly on the teacher's generated reasoning text
Implicit: injecting the teacher's hidden states into the student's layersduring training, via gated residual signals
Counterfactuals for evaluation and data augmentation are generated with a hardlabel-flip gate: a counterfactual is only accepted once the classifier's predictionhas demonstrably flipped, ensuring every accepted counterfactual carries genuineexplanatory value.

Results
Dataset	Baseline F1 (macro)	CoT-trained F1 (macro)
Loan default (LD1)	0.580	0.780
Clinical disposition (ER-REASON)	0.705	0.908
+20 F1 points on both benchmarks over classifiers trained on plain profiles
Student model is 36× smaller than the teacher LLM (110M vs 4B parameters)
Student training runs on consumer hardware (RTX 2070); the teacher is onlyneeded once, on an H100
CoT-trained students also generalize better to counterfactual scenarios
Repository Structure
├── configs/ # experiment configs (models, CoT, counterfactuals, sweeps)
├── prompts/ # system prompts for CoT and counterfactual generation
├── scripts/
│ ├── data/ # dataset preparation and processing
│ ├── generation/ # CoT and counterfactual generation pipelines
│ ├── training/ # student model training (explicit + implicit transfer)
│ ├── experiments/ # experiment runners and evaluation sweeps
│ └── utils/ # shared script utilities
├── src/
│ ├── cot/ # CoT generation and validation logic
│ ├── counterfactual/ # label-flip gated counterfactual pipeline
│ ├── data/ # data loading and handling
│ ├── data_preprocessing/ # preprocessing for training inputs
│ └── models/ # teacher and student model implementations
├── test/ # tests
└── utils/ # shared utilities

## Running the Code

The full pipeline setup is environment-specific (teacher inference requires an H100-class
GPU) and not intended for out-of-the-box use. Instead, each script in `scripts/` contains
detailed documentation explaining its role in the pipeline, its inputs and outputs, and
the configuration it expects. Start there for a guided tour of the framework.

Dependencies are listed in `requirements.txt`.
