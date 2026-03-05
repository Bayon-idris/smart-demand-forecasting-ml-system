from pathlib import Path
import sys

from pathlib import Path

def verify_data_path(file_path: str):
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    
    if not path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")
    
    return {"status": "success", "message": f"File {file_path} is ready"}

