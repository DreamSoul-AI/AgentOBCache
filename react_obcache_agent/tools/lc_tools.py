import ast
import operator
from typing import Any

from langchain_core.tools import tool

from tools.paper_tool import PaperSearchTool
from tools.wiki_tool import WikipediaTool


_WIKI_TOOL = WikipediaTool()
_PAPER_TOOL = PaperSearchTool()


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return a concise page summary."""
    return _WIKI_TOOL.search(query)


@tool
def wiki_lookup(keyword: str) -> str:
    """Look up a keyword in the current Wikipedia page and return the next matching sentence."""
    return _WIKI_TOOL.lookup(keyword)


@tool
def paper_search(query: str) -> str:
    """Search scholarly paper metadata and return the title, ordered authors, and year."""
    return _PAPER_TOOL.search(query)


def reset_tool_state() -> None:
    """Reset stateful tools before starting a new independent task."""
    _WIKI_TOOL.reset()


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression with numbers and arithmetic operators."""
    try:
        return str(_safe_eval(expression))
    except Exception as exc:
        return f"Calculator error: {exc}"


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operand = _eval_node(node.operand)
        return _ALLOWED_UNARYOPS[type(node.op)](operand)

    raise ValueError("unsupported calculator expression")


TOOLS = [wiki_search, wiki_lookup, paper_search, calculator]
TOOL_BY_NAME = {tool_item.name: tool_item for tool_item in TOOLS}
