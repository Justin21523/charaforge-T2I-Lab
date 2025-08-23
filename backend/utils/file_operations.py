# backend/utils/file_operations.py
import os
from pathlib import Path
from typing import List, Union


def list_directory(path: str = ".", max_items: int = 20) -> Union[List[str], str]:
    """
    List contents of a directory (with safety restrictions)

    Args:
        path: Directory path to list
        max_items: Maximum number of items to return

    Returns:
        List of file/directory names or error message
    """
    try:
        # Convert to Path object for safety
        dir_path = Path(path).resolve()

        # Safety check - only allow relative paths and specific directories
        allowed_dirs = [
            Path.cwd(),
            Path(os.getenv("AI_CACHE_ROOT", "/tmp")) / "outputs",
        ]

        if not any(
            str(dir_path).startswith(str(allowed_dir)) for allowed_dir in allowed_dirs
        ):
            return "Error: Access to this directory is not allowed"

        if not dir_path.exists():
            return f"Error: Directory '{path}' does not exist"

        if not dir_path.is_dir():
            return f"Error: '{path}' is not a directory"

        # List directory contents
        items = []
        for item in dir_path.iterdir():
            if len(items) >= max_items:
                items.append("... (more items)")
                break

            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                size_str = f"({size} bytes)" if size < 1024 else f"({size//1024} KB)"
                items.append(f"📄 {item.name} {size_str}")

        return items

    except PermissionError:
        return "Error: Permission denied"
    except Exception as e:
        return f"Error: {str(e)}"


def read_text_file(file_path: str, max_length: int = 2000) -> str:
    """
    Read content from a text file (with safety restrictions)

    Args:
        file_path: Path to the text file
        max_length: Maximum content length to return

    Returns:
        File content or error message
    """
    try:
        # Convert to Path object for safety
        file_path_obj = Path(file_path).resolve()

        # Safety check - only allow specific file types and locations
        allowed_extensions = {".txt", ".md", ".json", ".yaml", ".yml", ".log"}
        if file_path_obj.suffix.lower() not in allowed_extensions:
            return f"Error: File type '{file_path_obj.suffix}' is not allowed"

        # Check if file is in allowed directories
        allowed_dirs = [
            Path.cwd(),
            Path(os.getenv("AI_CACHE_ROOT", "/tmp")) / "outputs",
        ]

        if not any(
            str(file_path_obj).startswith(str(allowed_dir))
            for allowed_dir in allowed_dirs
        ):
            return "Error: Access to this file location is not allowed"

        if not file_path_obj.exists():
            return f"Error: File '{file_path}' does not exist"

        if not file_path_obj.is_file():
            return f"Error: '{file_path}' is not a file"

        # Read file content
        with open(file_path_obj, "r", encoding="utf-8") as f:
            content = f.read(max_length)

        if len(content) == max_length:
            content += "\n... (content truncated)"

        return content

    except UnicodeDecodeError:
        return "Error: File is not a valid text file"
    except PermissionError:
        return "Error: Permission denied"
    except Exception as e:
        return f"Error: {str(e)}"
