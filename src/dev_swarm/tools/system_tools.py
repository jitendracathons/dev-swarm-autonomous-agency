import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class FileWriteInput(BaseModel):
    path: str = Field(..., description="The full path including filename to write to (e.g., 'project_output/src/main.py').")
    content: str = Field(..., description="The code or text content to save.")

class AdvancedFileWriterTool(BaseTool):
    name: str = "Advanced FileWriter"
    description: str = "Writes content to a file. Automatically creates directories if they don't exist. Use this for all file creation."
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, path: str, content: str) -> str:
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote file to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"