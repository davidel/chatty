import os
import re
import ast
import json
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
    self.cache_path = os.path.join(self.sandbox_dir, ".chatty", "symbol_cache.json")
    self.cache = self._load_cache()

  def get_outline(self, rel_path: str) -> List[Dict[str, Any]]:
    """Extracts symbols using the best available provider with caching."""
    abs_path = os.path.abspath(os.path.join(self.sandbox_dir, rel_path))
    if not os.path.exists(abs_path):
      return []
    try:
      mtime = os.path.getmtime(abs_path)
    except OSError:
      mtime = 0.0
    cached_entry = self.cache.get(rel_path)
    if cached_entry and cached_entry.get("mtime") == mtime:
      return cached_entry.get("symbols", [])
    symbols = self._extract_symbols_uncached(abs_path)
    self.cache[rel_path] = {
      "mtime": mtime,
      "symbols": symbols
    }
    self._save_cache()
    return symbols

  def build_global_index(self) -> Dict[str, List[Dict[str, Any]]]:
    """Scans all non-ignored files in sandbox, updates cache, and returns the global index."""
    from chatty.safety import load_ignore_patterns, is_path_ignored
    ignore_patterns = load_ignore_patterns(self.sandbox_dir)
    global_index = {}
    for root, dirs, files in os.walk(self.sandbox_dir):
      for d in list(dirs):
        dir_path = os.path.relpath(os.path.join(root, d), self.sandbox_dir)
        if is_path_ignored(dir_path, ignore_patterns, is_dir=True):
          dirs.remove(d)
      for file in files:
        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, self.sandbox_dir)
        if is_path_ignored(rel_path, ignore_patterns):
          continue
        ext = os.path.splitext(file)[1].lower()
        if ext in (".py", ".js", ".jsx", ".ts", ".tsx", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java", ".go", ".rs"):
          symbols = self.get_outline(rel_path)
          if symbols:
            global_index[rel_path] = symbols
    return global_index

  def find_symbol(self, name: str) -> List[Dict[str, Any]]:
    """Searches the global index for any symbol matching the given name (case-insensitive)."""
    self.build_global_index()
    matches = []
    name_lower = name.lower()
    for rel_path, entry in self.cache.items():
      for sym in entry.get("symbols", []):
        if sym["name"].lower() == name_lower:
          matches.append({
            "name": sym["name"],
            "type": sym["type"],
            "line": sym["line"],
            "path": rel_path,
            "parent": sym.get("parent")
          })
    return matches

  def _load_cache(self) -> Dict[str, Any]:
    if os.path.exists(self.cache_path):
      try:
        with open(self.cache_path, "r", encoding="utf-8") as f:
          return json.load(f)
      except Exception:
        pass
    return {}

  def _save_cache(self):
    try:
      os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
      with open(self.cache_path, "w", encoding="utf-8") as f:
        json.dump(self.cache, f, indent=2)
    except Exception:
      pass

  def _extract_symbols_uncached(self, abs_path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(abs_path)[1].lower()
    if self.lsp_client and hasattr(self.lsp_client, "is_ready_for") and self.lsp_client.is_ready_for(ext):
      symbols = self._get_outline_via_lsp(abs_path)
      if symbols is not None:
        return symbols
    if HAS_TREE_SITTER:
      symbols = self._get_outline_via_tree_sitter(abs_path, ext)
      if symbols is not None:
        return symbols
    if ext == ".py":
      return self._get_outline_via_python_ast(abs_path)
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
          "parent": self._find_parent_class_ts(node)
        })
      return symbols
    except Exception:
      return None

  def _find_parent_class_ts(self, node: Any) -> Optional[str]:
    curr = node.parent
    while curr:
      if curr.type in ("class_definition", "class_declaration", "class_specifier"):
        name_node = curr.child_by_field_name("name")
        if name_node:
          return name_node.text.decode("utf-8", errors="ignore") if hasattr(name_node, "text") else ""
        for child in curr.children:
          if child.type in ("identifier", "type_identifier"):
            return child.text.decode("utf-8", errors="ignore") if hasattr(child, "text") else ""
      elif curr.type == "impl_item":
        type_node = curr.child_by_field_name("type")
        if type_node:
          return type_node.text.decode("utf-8", errors="ignore") if hasattr(type_node, "type") else ""
        for child in curr.children:
          if child.type == "type_identifier":
            return child.text.decode("utf-8", errors="ignore") if hasattr(child, "type_identifier") else ""
      curr = curr.parent
    if node.type == "method_declaration":
      receiver = node.child_by_field_name("receiver")
      if receiver:
        return self._find_type_node_recursive(receiver)
    return None

  def _find_type_node_recursive(self, node: Any) -> Optional[str]:
    if node.type in ("type_identifier", "type_name"):
      return node.text.decode("utf-8", errors="ignore") if hasattr(node, "text") else ""
    for child in node.children:
      res = self._find_type_node_recursive(child)
      if res:
        return res
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
    elif lang_name == "go":
      return """
        (type_spec name: (type_identifier) @class)
        (method_declaration name: (field_identifier) @method)
        (function_declaration name: (identifier) @function)
      """
    elif lang_name == "rust":
      return """
        (struct_item name: (type_identifier) @class)
        (function_item name: (identifier) @function)
      """
    elif lang_name == "java":
      return """
        (class_declaration name: (identifier) @class)
        (method_declaration name: (identifier) @method)
      """
    return ""
