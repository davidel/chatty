import os
import re
import ast
from typing import List, Dict, Any, Optional

try:
  import tree_sitter
  import tree_sitter_languages
  HAS_TREE_SITTER = True
except ImportError:
  HAS_TREE_SITTER = False


class SymbolVisitor(ast.NodeVisitor):

  def __init__(self):
    self.symbols = []
    self.current_class = None

  def visit_ClassDef(self, node: ast.ClassDef):
    self.symbols.append({
      "name": node.name,
      "type": "class",
      "line": node.lineno,
      "parent": self.current_class
    })
    old_class = self.current_class
    self.current_class = node.name
    self.generic_visit(node)
    self.current_class = old_class

  def visit_FunctionDef(self, node: ast.FunctionDef):
    type_str = "method" if self.current_class else "function"
    self.symbols.append({
      "name": node.name,
      "type": type_str,
      "line": node.lineno,
      "parent": self.current_class
    })
    self.generic_visit(node)


class SymbolExtractor:

  def __init__(self, sandbox_dir: str, lsp_client: Optional[Any] = None):
    self.sandbox_dir = sandbox_dir
    self.lsp_client = lsp_client

  def get_outline(self, rel_path: str) -> List[Dict[str, Any]]:
    """Extracts symbols using the best available provider."""
    abs_path = os.path.abspath(os.path.join(self.sandbox_dir, rel_path))
    if not os.path.exists(abs_path):
      return []
    ext = os.path.splitext(abs_path)[1].lower()
    # 1. Try LSP provider if client is active
    if self.lsp_client and hasattr(self.lsp_client, "is_ready_for") and self.lsp_client.is_ready_for(ext):
      symbols = self._get_outline_via_lsp(abs_path)
      if symbols is not None:
        return symbols
    # 2. Try Tree-Sitter provider if installed
    if HAS_TREE_SITTER:
      symbols = self._get_outline_via_tree_sitter(abs_path, ext)
      if symbols is not None:
        return symbols
    # 3. Fall back to Python AST for python files
    if ext == ".py":
      return self._get_outline_via_python_ast(abs_path)
    # 4. Final basic regex parser fallback
    return self._get_outline_via_regex(abs_path, ext)

  def _get_outline_via_lsp(self, abs_path: str) -> Optional[List[Dict[str, Any]]]:
    try:
      return self.lsp_client.get_document_symbols(abs_path)
    except Exception:
      return None

  def _get_outline_via_tree_sitter(self, abs_path: str, ext: str) -> Optional[List[Dict[str, Any]]]:
    lang_name = self._map_extension_to_language(ext)
    if not lang_name:
      return None
    try:
      with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        code_bytes = f.read().encode("utf-8")
      language = tree_sitter_languages.get_language(lang_name)
      parser = tree_sitter.Parser()
      parser.set_language(language)
      tree = parser.parse(code_bytes)
      query_str = self._get_tree_sitter_query(lang_name)
      if not query_str:
        return None
      query = language.query(query_str)
      captures = query.captures(tree.root_node)
      symbols = []
      for node, tag in captures:
        symbols.append({
          "name": node.text.decode("utf-8", errors="ignore") if hasattr(node, "text") else "",
          "type": tag,
          "line": node.start_point[0] + 1,
          "parent": None
        })
      return symbols
    except Exception:
      return None

  def _get_outline_via_python_ast(self, abs_path: str) -> List[Dict[str, Any]]:
    try:
      with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
      tree = ast.parse(code)
      visitor = SymbolVisitor()
      visitor.visit(tree)
      return visitor.symbols
    except Exception:
      return []

  def _get_outline_via_regex(self, abs_path: str, ext: str) -> List[Dict[str, Any]]:
    symbols = []
    try:
      with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
          match = re.search(r"^\s*(class|struct|def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
          if match:
            symbols.append({
              "name": match.group(2),
              "type": match.group(1),
              "line": idx,
              "parent": None
            })
    except Exception:
      pass
    return symbols

  def _map_extension_to_language(self, ext: str) -> Optional[str]:
    mapping = {
      ".py": "python",
      ".js": "javascript",
      ".jsx": "javascript",
      ".ts": "typescript",
      ".tsx": "typescript",
      ".cpp": "cpp",
      ".cc": "cpp",
      ".cxx": "cpp",
      ".h": "cpp",
      ".hpp": "cpp",
      ".java": "java",
      ".go": "go",
      ".rs": "rust"
    }
    return mapping.get(ext)

  def _get_tree_sitter_query(self, lang_name: str) -> str:
    if lang_name == "python":
      return """
        (class_definition name: (identifier) @class)
        (function_definition name: (identifier) @function)
      """
    elif lang_name in ("javascript", "typescript"):
      return """
        (class_declaration name: (identifier) @class)
        (function_declaration name: (identifier) @function)
        (method_definition name: (property_identifier) @method)
      """
    elif lang_name == "cpp":
      return """
        (class_specifier name: (type_identifier) @class)
        (function_definition declarator: (function_declarator declarator: (field_identifier) @method))
        (function_definition declarator: (function_declarator declarator: (identifier) @function))
      """
    return ""
