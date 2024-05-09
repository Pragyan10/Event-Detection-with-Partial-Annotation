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

###Pre Processing and getting the inputs ready for BERT
- This function constructs the necessary components for a BERT input example

def build_bert_example():

    is_impossible = (start_pos == -1)
    query_tokens = tokenizer.tokenize(query)

    # Calculate the number of tokens allowed for the context
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    # Tokenize the context into sub-tokens
    for (i, token) in enumerate(context):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        if len(all_doc_tokens) + len(sub_tokens) > max_tokens_for_doc:
            break

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # Convert start and end positions within the context to token indices
    tok_start_position = orig_to_tok_index[start_pos] if start_pos != -1 else -1
    tok_end_position = orig_to_tok_index[end_pos] if end_pos != -1 else -1

    tokens = [cls_token] + query_tokens + [sep_token]
    segment_ids = [cls_token_segment_id] * (len(query_tokens) + 2)

    # Append tokens from the document
    for i in range(len(all_doc_tokens)):
        tokens.append(all_doc_tokens[i])
        segment_ids.append(sequence_b_segment_id)

    tokens.append(sep_token)
    segment_ids.append(sequence_b_segment_id)

    # Convert tokens to IDs, prepare mask, and pad sequences
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0)
        segment_ids.append(pad_token_segment_id)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    # Compute offsets and positions
    doc_offset = len(query_tokens) + 2
    start_position = tok_start_position + doc_offset if start_pos != -1 else cls_index
    end_position = tok_end_position + doc_offset if end_pos != -1 else cls_index

    return input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position


### The magical Uncertainity Interval
- This function leverages the BertForQuestionAnswering model for a novel application of uncertainty estimation. This enhances the scores for gold labels and also makes the score for other unlabelled tokens as maybe for futher learning iteration.

  def uncertainty_interval():
    model.train()
    result = []
    with torch.no_grad():
        data_x, data_x_mask, data_segment_id, data_start, data_end, appendix = batch
        inputs = {
            'input_ids': data_x,
            'attention_mask':  data_x_mask,
            'token_type_ids':  data_segment_id,
            'start_positions': data_start,
            'end_positions':   data_end
        }
        for i in range(N):
            loss, start, end = model(**inputs)
            result.append([start, end])

    for elem in result:
        elem[0] = torch.argmax(elem[0], 1).detach().cpu().numpy()
        elem[1] = torch.argmax(elem[1], 1).detach().cpu().numpy()

    result = np.asarray(result)
    result = np.transpose(result, (2, 1, 0))

    start = result[:,0,:]
    end = result[:,1,:]

    start = [uncertainty_select(x) for x in start]
    end = [uncertainty_select(x) for x in end]

    start = torch.LongTensor(start).to(data_x.device)
    end = torch.LongTensor(end).to(data_x.device)

    inputs = {
        'input_ids': data_x,
        'attention_mask':  data_x_mask,
        'token_type_ids':  data_segment_id,
        'start_positions': start,
        'end_positions':   end
    }

    outputs = model(**inputs)
    return outputs



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


This innovative approach to event detection using partial annotations marks a significant advancement for not just event detection but also task including event argument detection and entity relation extraction. By effectively addressing the challenges associated with incomplete data, this method not only improves the accuracy and robustness of event detection models but also paves the way for future research and applications that can benefit from these methodological improvements in handling partial annotations.

## Reference 
[1]. Liu, Jian, et al. "Learning with Partial Annotations for Event Detection." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023.
https://aclanthology.org/2023.acl-long.30/

## **Notes**
This blog is made with the understanding of the paper and also based on the github link presented in the paper. It is recommended the reader view and paper and draw more insight on this idea. Link to the paper in Reference [1]. All copyright and credit goes to the authors of the paper. 

