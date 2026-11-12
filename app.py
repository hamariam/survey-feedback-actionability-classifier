
# Import required packages
import torch
import gradio as gr
import pandas as pd
import tempfile

from typing import Dict
from transformers import pipeline


# 1. Load the Hugging Face model
MODEL_ID = "mriamah/hf_survey-feedback-actionability"

classifier_pipeline = pipeline(
    task="text-classification",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None
)


# 2. Single-response classification function
def actionable_not_actionable_classifier(text: str) -> Dict[str, float]:

    # Check that the user entered a response
    if not text or not text.strip():
        raise gr.Error("Please enter a survey response.")

    # Get predictions from the model
    outputs = classifier_pipeline(text)[0]

    # Format predictions as a dictionary
    output_dict = {}

    for item in outputs:
        output_dict[item["label"]] = item["score"]

    return output_dict


# 3. Read uploaded CSV and identify its columns
def load_csv(file_path):

    if file_path is None:
        return (
            None,
            gr.Dropdown(choices=[], value=None),
            "Please upload a CSV file."
        )

    try:
        # Read the uploaded CSV
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")

        # Check whether the CSV contains data
        if df.empty:
            return (
                None,
                gr.Dropdown(choices=[], value=None),
                "The uploaded CSV is empty."
            )

        # Get the available column names
        columns = df.columns.tolist()

        return (
            df,
            gr.Dropdown(
                choices=columns,
                value=columns[0],
                interactive=True
            ),
            f"CSV loaded successfully: {len(df)} rows and {len(columns)} columns."
        )

    except Exception as e:
        return (
            None,
            gr.Dropdown(choices=[], value=None),
            f"Error reading CSV: {str(e)}"
        )


# 4. Classify all responses in the selected CSV column
def analyze_csv(file_path, feedback_column):

    if file_path is None:
        raise gr.Error("Please upload a CSV file first.")

    if not feedback_column:
        raise gr.Error("Please select the survey feedback column.")

    try:
        # Read the uploaded CSV
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")

        # Confirm that the selected column exists
        if feedback_column not in df.columns:
            raise gr.Error("The selected feedback column was not found.")

        # Identify rows containing actual survey feedback
        valid_mask = df[feedback_column].notna() & (
            df[feedback_column].astype(str).str.strip() != ""
        )

        valid_indices = df.index[valid_mask].tolist()

        # Extract text from the selected feedback column
        texts = df.loc[
            valid_indices,
            feedback_column
        ].astype(str).tolist()

        if not texts:
            raise gr.Error("No survey responses were found in that column.")

        # Use the same classification function as the Single Response tab
        predictions = [
            actionable_not_actionable_classifier(text)
            for text in texts
        ]

        # Prepare output columns
        df["AI Classification"] = "No response"
        df["Confidence Score"] = None

        actionable_count = 0
        non_actionable_count = 0

        # Process each prediction
        for index, prediction in zip(valid_indices, predictions):

            # Find the label with the highest confidence
            label = max(prediction, key=prediction.get)

            # Get its confidence score
            confidence = prediction[label]

            # Add predictions to the original spreadsheet
            df.at[index, "AI Classification"] = label
            df.at[index, "Confidence Score"] = round(confidence, 4)

            # Count actionable and non-actionable responses
            if label == "actionable":
                actionable_count += 1
            else:
                non_actionable_count += 1

        # Calculate summary metrics
        total_responses = len(texts)

        actionability_rate = (
            actionable_count / total_responses
        ) * 100

        # Create the analysis summary
        summary = f"""
## Survey Analysis Results

**Total responses analyzed:** {total_responses}

**Actionable responses:** {actionable_count}

**Non-actionable responses:** {non_actionable_count}

**Actionability rate:** {actionability_rate:.1f}%
"""

        # Create a downloadable CSV file
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        )

        output_file.close()

        df.to_csv(output_file.name, index=False)

        # Return the results, summary, and download file
        return df, summary, output_file.name

    except gr.Error:
        raise

    except Exception as e:
        raise gr.Error(f"An error occurred during analysis: {str(e)}")


# 5. Create the Gradio application with two tabs
with gr.Blocks() as demo:

    gr.Markdown("# 📝 Survey Feedback Actionability Classifier")

    gr.Markdown(
        "Use AI to identify actionable suggestions and opportunities "
        "for improvement in open-ended survey feedback."
    )

    with gr.Tabs():

        # TAB 1: Single response classification
        with gr.Tab("Single Response"):

            gr.Markdown("### Classify an individual survey response")

            single_input = gr.Textbox(
                lines=4,
                label="Survey Feedback",
                placeholder="Enter a survey response here..."
            )

            single_output = gr.Label(
                num_top_classes=2,
                label="Classification Results"
            )

            single_button = gr.Button("Classify Response")

            single_button.click(
                fn=actionable_not_actionable_classifier,
                inputs=single_input,
                outputs=single_output
            )

            # Example survey responses
            gr.Examples(
                examples=[
                    ["I would prefer having more time between conference sessions."],
                    ["The speakers were knowledgeable and engaging."],
                    ["The organization should offer virtual attendance options."],
                    ["I really enjoyed attending this year's conference."]
                ],
                inputs=single_input
            )

        # TAB 2: Batch CSV analysis
        with gr.Tab("Batch CSV Analysis"):

            gr.Markdown("### Analyze an entire survey spreadsheet")

            gr.Markdown(
                "Upload a CSV containing survey responses. "
                "Your original columns will be preserved."
            )

            gr.Markdown(
                "**Privacy reminder:** Do not upload identifiable "
                "or confidential information."
            )

            # Upload CSV file
            csv_input = gr.File(
                label="Upload Survey CSV",
                file_types=[".csv"],
                type="filepath"
            )

            upload_status = gr.Markdown()

            # Select the column containing survey feedback
            feedback_column = gr.Dropdown(
                choices=[],
                label="Select the column containing survey feedback",
                interactive=False
            )

            # Display uploaded spreadsheet
            preview = gr.Dataframe(
                label="Spreadsheet Preview",
                interactive=False
            )

            csv_input.change(
                fn=load_csv,
                inputs=csv_input,
                outputs=[
                    preview,
                    feedback_column,
                    upload_status
                ]
            )

            # Analyze all survey responses
            analyze_button = gr.Button(
                "Analyze Survey Responses",
                variant="primary"
            )

            # Display summary metrics
            summary_output = gr.Markdown(
                label="Analysis Summary"
            )

            # Display classified spreadsheet
            results_output = gr.Dataframe(
                label="Classified Survey Results",
                interactive=False
            )

            # Download classified CSV
            download_output = gr.File(
                label="Download Classified CSV"
            )

            analyze_button.click(
                fn=analyze_csv,
                inputs=[
                    csv_input,
                    feedback_column
                ],
                outputs=[
                    results_output,
                    summary_output,
                    download_output
                ]
            )


# 6. Launch the application
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
