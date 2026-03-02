from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import PyPDF2


class PDFExtractorToolInput(BaseModel):
    """Input schema for PDFExtractorTool."""
    file_path: str = Field(..., description="The absolute path to the PDF file to extract text from.")


class PDFExtractorTool(BaseTool):
    name: str = "PDF Text Extractor"
    description: str = (
        "A tool that extracts text content from a PDF file. "
        "Provide the absolute file path to the PDF document to get its text."
    )
    args_schema: Type[BaseModel] = PDFExtractorToolInput

    def _run(self, file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

            if not text.strip():
                return "No text could be extracted from the PDF. It might be an image-based PDF."

            return text
        except FileNotFoundError:
            return f"Error: The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred while reading the PDF: {str(e)}"
