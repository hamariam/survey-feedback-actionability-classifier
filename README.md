
# 📝 Survey Feedback Actionability Classifier

An AI-powered application that classifies open-ended survey feedback as actionable or non-actionable, helping organizations identify suggestions and opportunities for improvement.

## Project Overview

Organizations collect valuable qualitative feedback through conferences, workshops, professional development programs, and other initiatives. However, manually reviewing and categorizing large volumes of open-ended responses can be time-consuming.

This project explores how natural language processing (NLP) can help streamline that process by identifying survey responses that contain actionable suggestions or opportunities for improvement.

The application supports both individual response classification and batch analysis of survey results through CSV uploads.

## Features

### 1. Single Response Classification
- Enter an individual survey response.
- Receive an actionable or non-actionable classification.
- View model confidence scores.

### 2. Batch CSV Analysis
- Upload an existing survey CSV file.
- Select the column containing open-ended feedback.
- Classify multiple responses automatically.
- Preserve original spreadsheet columns.
- Add AI classification and confidence score columns.

### 3. Summary Metrics
- Total responses analyzed.
- Number of actionable responses.
- Number of non-actionable responses.
- Overall actionability rate.

### 4. Downloadable Results
Export the original survey spreadsheet with AI-generated classifications and confidence scores added.

## Classification Examples

| Survey Feedback | Classification |
|---|---|
| The registration process should be simplified. | Actionable |
| I would prefer more time for networking. | Actionable |
| The organization should offer virtual attendance options. | Actionable |
| The speakers were knowledgeable and engaging. | Non-actionable |
| I really enjoyed attending the conference. | Non-actionable |

**Actionable:** A response containing a suggestion, recommendation, request, or identifiable opportunity for improvement.

**Non-actionable:** General opinions, positive feedback, or personal experiences without a specific suggested improvement.

## Technology Stack

- Python
- Hugging Face Transformers
- DistilBERT
- PyTorch
- Gradio
- Pandas
- Hugging Face Spaces

## Model

The application uses a fine-tuned DistilBERT model based on:

`distilbert/distilbert-base-uncased`

[View the trained model](https://huggingface.co/mriamah/hf_survey-feedback-actionability)

## Dataset

The model was trained using a synthetic dataset of 100 labeled survey feedback responses.

The dataset contains two classification labels:

- `actionable`
- `non_actionable`

[View the dataset](https://huggingface.co/datasets/mriamah/survey-feedback-actionability)

## Sample Data

A sample CSV is provided in the `sample-data` folder so users can test the batch analysis functionality.

The sample includes multiple columns, demonstrating how users can select the appropriate feedback column without modifying their original spreadsheet.

[Download the sample survey feedback CSV](sample-data/survey_feedback_test.csv)

## How to Use

1. Open the live application.
2. Choose Single Response to classify an individual comment, or Batch CSV Analysis to process a spreadsheet.
3. For CSV analysis, upload a CSV file.
4. Select the column containing survey feedback.
5. Run the analysis.
6. Review the classification results and summary metrics.
7. Download the classified CSV.

## Live Demo

[Try the Survey Feedback Actionability Classifier](https://huggingface.co/spaces/mriamah/survey-feedback-actionability-classifier)

## Limitations

This is an experimental portfolio project developed using a relatively small synthetic dataset of 100 survey responses.

Model confidence scores do not guarantee classification accuracy. The model may misclassify responses, particularly when feedback contains indirect suggestions or nuanced context.

Human review is recommended before using classifications to inform organizational decisions.

The application is intended as a demonstration of applied NLP and business analytics, not as a production-ready decision-making system.

## Project Purpose

This project explores the practical application of AI and NLP to business data analytics, with an emphasis on transforming qualitative feedback into structured information that can support organizational evaluation and improvement.
