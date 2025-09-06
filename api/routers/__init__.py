# api/routers/__init__.py - Router package initialization
"""
API Routers package for CharaForge T2I Lab

This package contains all API route handlers organized by functionality.
Each router module handles a specific aspect of the API.
"""

# Import routers with error handling to prevent startup failures
import sys

# Health router - core functionality
try:
    from . import health

    __all__ = ["health"]
except ImportError as e:
    print(f"Warning: Failed to import health router: {e}", file=sys.stderr)
    __all__ = []

# T2I router - image generation
try:
    from . import t2i

    __all__.append("t2i")
except ImportError as e:
    print(f"Warning: Failed to import t2i router: {e}", file=sys.stderr)

# Fine-tuning router - model training
try:
    from . import finetune

    __all__.append("finetune")
except ImportError as e:
    print(f"Warning: Failed to import finetune router: {e}", file=sys.stderr)

# Batch processing router
try:
    from . import batch

    __all__.append("batch")
except ImportError as e:
    print(f"Warning: Failed to import batch router: {e}", file=sys.stderr)

# Export router
try:
    from . import export

    __all__.append("export")
except ImportError as e:
    print(f"Warning: Failed to import export router: {e}", file=sys.stderr)

# Safety router
try:
    from . import safety

    __all__.append("safety")
except ImportError as e:
    print(f"Warning: Failed to import safety router: {e}", file=sys.stderr)

# Monitoring router
try:
    from . import monitoring

    __all__.append("monitoring")
except ImportError as e:
    print(f"Warning: Failed to import monitoring router: {e}", file=sys.stderr)

print(f"Loaded routers: {', '.join(__all__)}")
