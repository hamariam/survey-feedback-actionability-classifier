
# Import the required packages
import torch
import gradio as gr
import os

from typing import Dict
from transformers import pipeline

# 1. Define the Hugging Face model repository
MODEL_ID = "mriamah/hf_survey-feedback-actionability"

# 2. Load the model and tokenizer once when the app starts
classifier_pipeline = pipeline(
    task="text-classification",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None
)

# 3. Define the classification function
def actionable_not_actionable_classifier(text: str) -> Dict[str, float]:

    # Get predictions from the model
    outputs = classifier_pipeline(text)[0]

    # Format predictions as a dictionary
    output_dict = {}

    for item in outputs:
        output_dict[item["label"]] = item["score"]

    # Return the formatted predictions
    return output_dict


# 4. Create the Gradio interface
demo = gr.Interface(
    fn=actionable_not_actionable_classifier,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter a survey response here..."
    ),
    outputs=gr.Label(num_top_classes=2),
    title="📝 Survey Feedback Actionability Classifier",
    description=(
        "Use AI to analyze open-ended survey feedback and identify "
        "responses that contain actionable suggestions or opportunities "
        "for improvement. Enter a survey response to classify it as "
        "actionable or non-actionable. This tool supports survey analysis "
        "and does not replace human review."
    ),
    examples=[
        ["I would prefer having more time between conference sessions."],
        ["The speakers were knowledgeable and engaging."],
        ["The organization should offer virtual attendance options."],
        ["I really enjoyed attending this year's conference."]
    ]
)

# 5. Launch the application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
