# Reinforcement Learning from Human Feedback (RLHF) on Custom Data

This project implements Reinforcement Learning from Human Feedback (RLHF) on custom data, replicating the pipeline described in the paper "Learning to Summarize from Human Feedback" by OpenAI. The RLHF process involves three key steps: Supervised Fine-Tuning (SFT), Reward Model Training, and Policy Optimization, enabling the model to generate summaries that align with human preferences.

<p align="center">
  <img src="https://github.com/ranzeet013/RLHF-CustomData/blob/main/assets/Screenshot%202025-03-22%20at%2010.35.33.png" alt="RLHF Pipeline from "Learning to Summarize from Human Feedback" width="700">
</p>
---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Contributions of the Paper](#key-contributions-of-the-paper)
- [Why RLHF?](#why-rlhf)
- [Implementation Details](#implementation-details)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Reward Model Training](#reward-model-training)
  - [Policy Optimization](#policy-optimization)
  - [Judger](#Judger)
- [Workflow](#Workflow)
- [Results](#results)
- [Citations](#citations)
- [License](#license)

---

## Project Overview

This project implements Reinforcement Learning from Human Feedback (RLHF) on custom data, replicating the pipeline described in the paper **"Learning to Summarize from Human Feedback"** by OpenAI. The RLHF process involves three key steps:

1. **Supervised Fine-Tuning (SFT)**: Fine-tune a base model on human-labeled response data to ensure it generates coherent and relevant responses.
2. **Reward Model Training**: Train a reward model to predict human preferences, enabling the model to distinguish between high-quality and low-quality responses.
3. **Policy Optimization**: Use reinforcement learning (e.g., PPO) to fine-tune the SFT model using the reward model, aligning the model's outputs with human preferences.

---

## Key Contributions of the Paper
- **Human Feedback on Custom Data**: Leverages human preferences specific to your dataset to guide the training of response generation models, ensuring the outputs align with domain-specific requirements.
- **Reinforcement Learning with PPO**: Uses Proximal Policy Optimization (PPO) to fine-tune the model on custom data, aligning its outputs with human preferences while maintaining stability during training.
- **Scalability to Custom Domains**: Demonstrates that RLHF can be effectively applied to custom datasets, enabling the fine-tuning of large language models like Flan-T5 for domain-specific generations tasks.

---

## Why RLHF?
Traditional supervised learning for summarization relies on human-labeled datasets, but it has limitations:

- **Lack of Preference Diversity**: Supervised learning assumes a single "correct" response, ignoring the diversity of human preferences and the fact that multiple valid responses may exist for the same input.
- **Static Training**: Once trained, the model cannot adapt to new or evolving human preferences without additional labeled data.

#### RLHF addresses these limitations by:
- **Using a Reward Model**: A reward model is trained to capture human preferences, allowing the system to evaluate and rank responses based on their alignment with what humans find useful or high-quality.
- **Fine-Tuning with Reinforcement Learning**: The model is fine-tuned using reinforcement learning (e.g., PPO) to maximize rewards from the reward model. This ensures the model dynamically adapts to human preferences and generates responses that are not only accurate but also aligned with what users value.
- **Handling Ambiguity**: RLHF can handle ambiguous or subjective tasks (like summarization, question-answering) by learning from diverse human feedback rather than relying on a single "correct" answer.
---

## Implementation Details
### 1. Supervised Fine-Tuning (SFT)
Supervised Fine-Tuning (SFT) is the process of fine-tuning a pre-trained language model on a labeled dataset using supervised learning. It is a critical step in many machine learning pipelines, including Reinforcement Learning from Human Feedback (RLHF), to ensure the model performs well on a specific task before further optimization.

- **Base Model**: `Flan-T5 (google/flan-t5-base)`, a powerful sequence-to-sequence model fine-tuned for a variety of tasks.
- **Custom Dataset**: A domain-specific response dataset with human-labeled examples, tailored to your specific use case (e.g., legal documents, medical reports, or technical articles)

#### Implementation:
- The base model is fine-tuned using supervised learning on response custom dataset of human-labeled responses examples.
- The fine-tuned model is saved as the **SFT model**.

#### Purpose:
- Ensures the model learns to generate coherent, relevant, and domain-specific summaries before reinforcement learning.
- Provides a strong foundation for the subsequent reward modeling and policy optimization steps.

#### Why SFT?
Supervised Fine-Tuning is the critical first step in the RLHF pipeline. It ensures that the model:

- **Understands the Domain**: Learns the specific language, structure, and nuances of custom dataset.
- **Generates Reasonable Summaries**: Produces meaningful and contextually appropriate outputs, which are essential for effective reward modeling.
- **Avoids Poor Initial Outputs**: Prevents the model from generating irrelevant or incoherent rsponses during reinforcement learning, which could destabilize training.
---

### 2. Reward Model Training

Reward Modeling is the process of training a model to predict human preferences for different outputs (e.g., summaries, responses, or actions). It assigns a scalar reward to each output, reflecting how well it aligns with what humans find useful, high-quality, or desirable. Reward modeling is a critical component of Reinforcement Learning from Human Feedback (RLHF), as it provides the feedback signal needed to guide the training of a policy model.

- **Base Model**: `Flan-T5 (google/flan-t5-base)`, a powerful sequence-to-sequence model adapted for reward prediction.
- **Custom Dataset**:  A dataset of human preferences specific to your task (e.g., pairs of summaries with rankings or scores, where humans have indicated which summary is better).

#### Implementation:
- The reward model is trained to predict human preferences using a **ranking loss** (e.g., pairwise ranking) or **regression loss** (e.g., mean squared error for scalar scores).
- The trained model is saved as the **reward model**.

#### Purpose:
- Provides a **scalar reward** to evaluate the quality of responsess generated by the policy model.
- Guides the policy optimization process by quantifying how well a response aligns with human preferences.

#### Why Reward Modeling?
Reward modeling is the process of training a model to predict human preferences for different outputs (e.g., summaries or responses). It assigns a scalar reward to each output, reflecting its alignment with what humans find useful or high-quality. This step is crucial for enabling reinforcement learning from human feedback (RLHF).It ensures that the model:

- **Captures Human Preferences**: Learns the specific language, structure, and nuances on custom dataset.
- **Provides Feedback for RL**: Supplies a s**calar reward** signal to guide the **policy model** during **reinforcement learning**.
- **Handles Subjectivity**: Adapts to diverse human preferences, ensuring outputs align with real-world user needs.
---

### 3. Policy Optimization
**Policy Optimization** is the process of fine-tuning a model using **reinforcement learning** to maximize a reward signal. In the context of **Reinforcement Learning from Human Feedback (RLHF)**, policy optimization ensures that the model's outputs (e.g., summaries, responses) align with **human preferences** by using feedback from a **reward model**. It is the final step in the RLHF pipeline and is crucial for adapting the model to produce high-quality, human-aligned outputs.

- **SFT Model**: The fine-tuned model from Step 1, which generates coherent and domain-specific responses.
- **Reward Model**: The trained reward model from Step 2, which provides scalar rewards based on human preferences.

#### Implementation:
- The **SFT model** is used as the **initial policy model**.
- Responses (**trajectories**) are generated using the policy model.
- Rewards are computed for these responses using the **reward model**.
- he policy model is optimized using the **Proximal Policy Optimization (PPO)** algorithm to maximize rewards.

#### Purpose:
- Fine-tunes the model to generate responses that are **aligned with human preferences.**
- Ensures the model produces outputs that are not only coherent but also highly relevant and useful for your custom dataset.


#### Why Policy Optimization?
Policy Optimization is the **final step** in the RLHF pipeline. It uses **reinforcement learning** to fine-tune the model based on the **rewards** provided by the reward model. This ensures that the model's outputs are not only **coherent** but also **aligned with human preferences**.

- **Aligns with Human Preferences:**:The policy model learns to maximize rewards from the reward model, ensuring its outputs align with what humans find useful or high-quality. This step bridges the gap between the model's capabilities and human expectations.
- **Improves Output Quality**: By fine-tuning the model to maximize rewards, policy optimization ensures the generated responses are **relevant, concise, and domain-specific**.It refines the model's behavior to produce outputs that are more aligned with custom dataset's requirements.
- **Handles Complex Preferences:**: Human preferences can be complex and subjective, especially for tasks like summarization response generation.**Policy optimization** allows the model to adapt to these preferences dynamically, producing outputs that are broadly acceptable
- **Uses Reinforcement Learning**: The policy model is optimized using **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm.PPO ensures stable and efficient updates to the model while maximizing rewards.
- **Produces Human-Aligned Outputs**: The final policy model generates responses that are **coherent, high-quality, and aligned with human preferences**. This makes the model more useful and reliable for real-world applications.
---

### 4. Judger
The Judger is a critical component in the RLHF pipeline that evaluates responses (e.g., summaries or answers) and calculates preference scores based on human feedback. It acts as the bridge between human preferences and the reward model, ensuring the model's outputs align with what humans find useful or high-quality.

### What is the Judger?
#### 1.Defination
- The Judger is a system or model that evaluates responses and assigns preference scores based on human feedback. It uses a dataset of human preferences (e.g., pairs of responses with rankings or scores) to learn how to judge the quality of outputs.
#### 2.Purpose:
- To provide a scalar preference score for each response, reflecting its alignment with human preferences.To guide the reward model in learning how to predict human preferences accurately.
#### 3.Key Characteristics:
- **Preference-Based Evaluation**: The Judger learns from human feedback to distinguish between high-quality and low-quality responses.
- **Scalar Output**: Assigns a single score to each response, quantifying its alignment with human preferences.
- **Domain-Specific Adaptation**: Tailored to your custom dataset, ensuring it captures preferences relevant to your specific task.

#### Why is the Judger Important?

The Judger evaluates responses and calculates preference scores based on human feedback, ensuring the reward model can accurately predict human preferences and guide the policy optimization process. Without the Judger, the model would lack the critical feedback needed to align its outputs with human expectations, making it difficult to produce high-quality, user-aligned results.

- **Captures Human Preferences**: The Judger learns to evaluate responses based on what humans find useful or high-quality, ensuring the model aligns with real-world user needs. It handles the subjectivity of tasks like summarization, where multiple valid responses may exist.
- **Provides Feedback for Reward Modeling:**: The preference scores calculated by the Judger are used to train the reward model. This ensures the reward model can accurately predict human preferences and provide meaningful feedback during policy optimization.
- **Improves Output Quality**: By evaluating responses and assigning preference scores, the Judger helps refine the model's outputs to be more relevant, concise, and aligned with human expectations.

---

## Workflow 
Detailed workflow for training a model using Reinforcement Learning with Human Feedback (RLHF). The process consists of five key steps: preparing datasets, fine-tuning a model using Supervised Fine-Tuning (SFT), training a reward model, performing policy optimization, and evaluating the optimized model.

### 1. Prepare Your Custom Dataset

#### Key Points:

- **SFT Dataset:** Collect a dataset containing input texts and their corresponding human-written responses.
- **Reward Dataset:** Prepare a dataset of human preferences, such as pairs of responsess with rankings or scores.
- **Format:** Ensure datasets are in a compatible format such as JSON, CSV, or a Hugging Face Dataset to streamline processing.

#### Purpose:

This step ensures that both the supervised fine-tuning and reward model training have high-quality data to learn from.

### 2. Run the SFT Training Script

#### Key Points:

- **Fine-Tune the Base Model:** Train a base language model (e.g., Flan-T5, GPT, or LLaMA) using supervised learning on the SFT dataset.
- **Save the SFT Model:** Store the fine-tuned model for further training steps.
- **Purpose:** This step helps the model generate coherent and relevant responses before applying reinforcement learning.

#### Steps:

- Load the base model and tokenizer.
- Preprocess the dataset (tokenization, formatting, padding, etc.).
- Train the model using supervised learning.
- Save the fine-tuned model for later use.

### 3. Train the Reward Model

#### Key Points:

- **Preference-Based Training:** Train a reward model using human preference data, typically with a ranking loss (e.g., pairwise ranking) or a regression loss (e.g., scalar scores).
- **Save the Reward Model:** Store the trained reward model for use in policy optimization.

#### Purpose:

The reward model assigns a score to generated summaries, allowing reinforcement learning to optimize for human-aligned output.

#### Steps:

- Load a pre-trained model for reward modeling (e.g., RoBERTa, DistilBERT, or a similar transformer model).
- Process the preference dataset.
- Train the reward model using ranking loss or regression loss.
- Save the trained reward model.

### 4. Perform Policy Optimization

#### Key Points:

- **Initialize Policy Model:** Start with the SFT model as the initial policy model.
- **Generate and Evaluate Summaries:** Use the model to generate responses and evaluate them using the trained reward model.
- **Optimize with PPO:** Utilize the Proximal Policy Optimization (PPO) algorithm to fine-tune the policy model by maximizing rewards.

#### Steps:

- Load the fine-tuned SFT model.
- Generate responsess for input texts.
- Compute reward scores using the trained reward model.
- Apply PPO to optimize the policy model.
- Save the optimized model.

#### Purpose:

This step ensures the model improves its response generation ability by learning from human preferences.

### 5. Evaluate the Optimized Model

#### Key Points:

- **Human Evaluation:** Compare generated responses with human-written responsess to assess quality.
- **Automated Metrics:** Use evaluation metrics like ROUGE and BLEU to measure generation performance quantitatively.

#### Steps:

- Collect a test dataset with human-annotated responses.
- Generate responsess using the optimized model.
- Compare generated responsess with human-written ones.
- Compute evaluation metrics (e.g., ROUGE-L, BLEU, METEOR).
- Conduct human evaluations if possible.

#### Purpose:

This final step ensures the model produces high-quality, human-aligned responses for real-world applications.

---

### Summary of the Workflow:

| Step | Description |
|------|-------------|
| **Prepare Dataset** | Collect input-response pairs and preference data. |
| **Train SFT Model** | Fine-tune a base model with supervised learning. |
| **Train Reward Model** | Train a model to evaluate response quality based on human preferences. |
| **Policy Optimization** | Use PPO to optimize the response generation model based on reward scores. |
| **Evaluate Model** | Measure performance using human evaluation and automated metrics. |

---


### 3. Evaluate the Model
After training, evaluate the model to ensure it generates high-quality, human-aligned responsess. Use:

- **Human Evaluation:**: Compare model-generated responses with human-written ones to assess quality and alignment with preferences.
- **Automated Metrics**:  Use metrics like ROUGE or BLEU to objectively measure response performance.
- **Purpose**: Validate that the model meets the desired standards for coherence, relevance, and domain-specific accuracy.

## Results
- **SFT Model**: Achieved reasonable performance on the text grneration task, generating coherent and domain-specific responsess that serve as a strong foundation for further optimization.
- **Reward Model**: Successfully predicted human preferences for responsess, providing a reliable scalar reward signal to guide the policy optimization process.
- **Policy Optimization**:  Improved the quality of responsess using the PPO algorithm, ensuring the outputs are not only coherent but also aligned with human preferences.
- **Overall Impact**: The final model produces high-quality, human-aligned responsess tailored to custom dataset, demonstrating the effectiveness of the RLHF pipeline.

## Citations
If you use this code or reference the paper, please cite the following:

```bibtex
@article{stiennon2020learning,
  title={Learning to summarize from human feedback},
  author={Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano},
  journal={arXiv preprint arXiv:2009.01325},
  year={2020}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


