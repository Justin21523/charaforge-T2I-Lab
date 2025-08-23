# backend/utils/calculator.py
import math
import operator
from typing import Union


def calculate(expression: str) -> Union[float, str]:
    """
    Safely evaluate mathematical expressions

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        Result of the calculation or error message
    """
    try:
        # Define safe operations
        safe_dict = {
            "__builtins__": {},
            "__name__": "safe_eval",
            "__file__": "safe_eval",
            "__package__": None,
            # Math operators
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "//": operator.floordiv,
            "%": operator.mod,
            "**": operator.pow,
            "^": operator.xor,
            # Math functions
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
        }

        # Clean expression (remove extra spaces)
        expression = expression.strip()

        # Evaluate safely
        result = eval(expression, safe_dict, {})

        # Format result
        if isinstance(result, float):
            if result.is_integer():
                return int(result)
            else:
                return round(result, 6)

        return result

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {str(e)}"
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except Exception as e:
        return f"Error: {str(e)}"
