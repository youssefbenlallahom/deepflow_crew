#!/usr/bin/env python
import os
import sys
import warnings

from datetime import datetime

from deepflow_crew.crew import DeepflowCrew

# Build a portable path to the PDF — works on any machine
PDF_PATH = os.path.join(os.path.dirname(__file__), 'tools', 'COMPLAINT Doe 1 v Epstein.pdf')

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter your research topic: ").strip()

    if not topic:
        topic = "When and how did Jeffrey Epstein die?"

    inputs = {
        'topic': topic,
        'current_year': str(datetime.now().year),
        'pdf_path': PDF_PATH
    }
    
    try:
        DeepflowCrew().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "Please review the provided indictment document and find the exact addresses of Jeffrey Epstein's primary residences in New York and Palm Beach. Once you have those specific addresses, I want to know what happened to those properties after his death. Look up who bought them, how much they sold for, and what their current status or planned use is. Please read the full text of the most recent news articles about these real estate sales so you can give me a detailed and accurate update.",
        'current_year': str(datetime.now().year),
        'pdf_path': PDF_PATH
    }
    try:
        DeepflowCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DeepflowCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "Please review the provided indictment document and find the exact addresses of Jeffrey Epstein's primary residences in New York and Palm Beach. Once you have those specific addresses, I want to know what happened to those properties after his death. Look up who bought them, how much they sold for, and what their current status or planned use is. Please read the full text of the most recent news articles about these real estate sales so you can give me a detailed and accurate update.",
        "current_year": str(datetime.now().year),
        "pdf_path": PDF_PATH
    }
    
    try:
        DeepflowCrew().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
