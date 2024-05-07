# Advancing Event Detection with Partial Annotations: Breakthroughs and Methodologies

Event Detection (ED) is a critical task within the broader field of natural language processing, aimed at identifying and categorizing occurrences of specific events in textual data. Traditionally, ED relies heavily on fully labeled datasets for model training. However, these ideal conditions are rarely met in practical scenarios, leading to datasets with partial or incomplete annotations. This issue significantly hampers the training of effective models using standard supervised learning techniques, as they struggle to cope with the incomplete data, leading to poor generalization on real-world data.

For example in the figure below we see that <u> **"devastated"** </u> is considered as a False Negative since it is not annotated in the dataset and falls under <u> **Partial Annotation** </u>. This is just an instance and there are many dataset that we use which are only partially annotated. This becomes a big issue since Event Detection requires a fully annotated dataset to perform better.

![ImageForPartialAnnotation](/Images/PartialAnnotation.png)

## [1] Proposed Approach
To effectively handle the inherent challenges of partial annotations within event detection datasets, this work introduces an advanced method that incorporates contrastive learning techniques combined with an uncertainty-guided training mechanism. This dual approach aims to robustly distinguish between genuine event triggers and the surrounding contextual words, thus enhancing the model's accuracy and reliability in event categorization, even when faced with datasets that are only partially annotated or contain noisy labels.

![ImageForApproachOverview](/Images/ApproachOverview.png)

### Visual Explanation of the Trigger Localization

**Comparative Overview** <br />
This section contrasts traditional hard classification methods with our advanced trigger localization strategy. Unlike traditional methods that often misclassify due to unclear training data distinctions, our approach uses a contrastive learning framework that dynamically enhances the separation between true event triggers and non-trigger words. This method focuses on boosting scores for potential triggers while suppressing those for irrelevant contexts, thus improving the model’s focus and reducing the influence of noisy data.

<img src="/Images/TriggerLocalization.png" width="400" height="250" alt="TriggerLocalization" />

This visualization demonstrates how our model processes text to isolate and identify event triggers more effectively than traditional methods, providing a clear, operational insight into the advantages of contrastive learning in handling partial annotations.


### Uncertaininty-guided training mechanism

**Adaptive Training Through Confidence Assessment** <br />
Our uncertainty-guided training incorporates Monte Carlo Dropout to estimate the model's predictive uncertainty. By reinforcing high-confidence predictions and cautiously adjusting parameters for low-confidence ones, the model learns from its most reliable outputs and remains adaptable. This approach not only helps mitigate the impact of noisy or incomplete data but also enhances the model's generalization capabilities across diverse datasets.

## [2] Important Key Code Snippets

** To do ** [Need to fill this part] 


## [3] Key Results and Performance

The approach does very well compared to existing baselines in the event detection task. 
Some key points and results:
- **On Evaluation Settings**
  - **Full Training Setting:** Evaluates the model using the original training set. The proposed method achieves the best F1 scores, demonstrating its effectiveness.
  - **Data Removal Setting:** Studies the impact of reducing the number of positive examples. Results show the model consistently outperforms others, confirming that the lack of positive instances isn’t a major factor hindering learning when using pre-trained language models.
  - **Data Masking Setting:** Simulates a more severe partial annotation scenario by removing the labeling information of some events. The proposed method outperforms previous methods by significant margins, showcasing its ability to learn effectively from unlabeled data.
- On Baselines
<table>
<tr>
  <td>

  **ACE 2005 Results**
  | Model            | F1 on Original Set | F1 on Revised Set |
  |------------------|--------------------|-------------------|
  | **Hybrid (2016)** | 71.4               | 73.3              |
  | **SeqBERT (2019)** | 72.3               | 73.8              |
  | **BERTQA (2020)** | 72.4               | 74.5              |
  | **OneIE (2020)** | 74.7               | 75.3              |
  | **FourIE (2021)** | 74.9               | 75.3              |
  | **PromptLoc (Proposed)** | 73.9       | 76.6              |

  </td>
  <td>

  **MAVEN Results**
  | Model                    | F1 Score |
  |--------------------------|----------|
  | **Hybrid (2016)**        | 65.0     |
  | **OneIE (2021)**         | 66.4     |
  | **BERTQA (2020)**        | 66.3     |
  | **DMBERT (2019)**        | 67.2     |
  | **BERT-CRF (2020)**      | 67.8     |
  | **PromptLoc (Proposed)** | 68.9     |

  </td>
</tr>
</table>






## [4] Practical Examples and Real-World Applications

The utility of the approach on some test cases, demonstrating significant practical benefits in real-world applications. Here example sentences are used and are used for event prediction for a given event type. The input for the model is always an event type and a sentence where the events are to be predicted. The output is a label list of the event predicted. 

**Example 1:**
Text: "A man died when a heavy tank devastated the hotel." <br />
| Description       | A | man | died | when | a | heavy | tank | devastated | the | hotel |
|-------------------|---|-----|------|------|---|-------|------|------------|-----|-------|
| Gold Label        | O | O   | Die  | O    | O | O     | O    | Attack     | O   | O     |
| Partial Annotation| O | O   | Die  | O    | O | O     | O    | O          | O   | O     |
| Model Output      | O | O   | Die  | O    | O | O     | O    | Attack     | O   | O     |


Analysis: The model successfully identifies the "Attack" event, despite its absence in the partial annotations. <br />

**Example 2:**
Text: "The company announced a new investment in the technology sector." <br />
| Description       | The | company | announced | a  | new | investment | in | the | technology | sector |
|-------------------|-----|---------|-----------|----|-----|------------|----|-----|------------|--------|
| Gold Label        | O   | O       | O         | O  | New | Investment | O  | O   | O          | O      |
| Partial Annotation| O   | O       | O         | O  | O   | Investment | O  | O   | O          | O      |
| Model Output      | O   | O       | O         | O  | New | Investment | O  | O   | O          | O      |


Analysis: The model correctly fills in the missing "New" label, showing its capability to infer and complete partially labeled data accurately. <br />

**Some more examples: **

| Description       | She  | has  | been | offered  | a   | significant | role  | in  | the | upcoming | movie  |
|-------------------|------|------|------|----------|-----|-------------|-------|-----|-----|----------|--------|
| Gold Label        | O    | O    | O    | Offered  | O   | O           | O     | O   | O   | O        | O      |
| Partial Annotation| O    | O    | O    | O        | O   | O           | O     | O   | O   | O        | O      |
| Model Output      | O    | O    | O    | Offered  | O   | O           | O     | O   | O   | O        | O      |


| Description       | The  | wind | turbine | generates | more | energy | when | it's  | windy  |
|-------------------|------|------|---------|-----------|------|--------|------|-------|--------|
| Gold Label        | O    | O    | O       | Generates | O    | O      | O    | O     | O      |
| Partial Annotation| O    | O    | O       | O         | O    | O      | O    | O     | O      |
| Model Output      | O    | O    | O       | Generates | O    | O      | O    | O     | O      |


These examples illustrate the model's practical ability to correct and complete partial annotations, enhancing the reliability and accuracy of event detection in various texts compared to the existing hard classification.

## Conclusion
This innovative approach to event detection using partial annotations marks a significant advancement for not just event detection but also task including event argument detection and entity relation extraction. By effectively addressing the challenges associated with incomplete data, this method not only improves the accuracy and robustness of event detection models but also paves the way for future research and applications that can benefit from these methodological improvements in handling partial annotations.

## Reference 
[1]. Liu, Jian, et al. "Learning with Partial Annotations for Event Detection." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023.
https://aclanthology.org/2023.acl-long.30/

## **Notes**
This blog is made with the understanding of the paper and also based on the github link presented in the paper. It is recommended the reader view and paper and draw more insight on this idea. Link to the paper in Reference [1]. All copyright and credit goes to the authors of the paper. 

