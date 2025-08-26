# frontend/gradio_app/utils/gradio_utils.py
"""
Gradio Utility Functions
"""
import base64
import io
import tempfile
from pathlib import Path
from PIL import Image


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def base64_to_image(base64_str):
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None


def save_temp_image(image, suffix=".png"):
    """Save PIL Image to temporary file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        image.save(temp_file.name)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Error saving temp image: {e}")
        return None


def format_file_size(bytes_size):
    """Format file size in human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def format_duration(seconds):
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分鐘"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小時"


def validate_image_file(file_path):
    """Validate if file is a valid image"""
    if not file_path or not Path(file_path).exists():
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_error_message(error, context="操作"):
    """Create formatted error message"""
    return f"❌ {context}失敗: {str(error)}"


def create_success_message(message, context="操作"):
    """Create formatted success message"""
    return f"✅ {context}成功: {message}"


def create_warning_message(message, context="警告"):
    """Create formatted warning message"""
    return f"⚠️ {context}: {message}"


def create_info_message(message, context="信息"):
    """Create formatted info message"""
    return f"ℹ️ {context}: {message}"
