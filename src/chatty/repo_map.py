import ast
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("chatty")

try:
  import tree_sitter
  HAS_TREE_SITTER = True
except ImportError:
  HAS_TREE_SITTER = False


def get_tree_sitter_language(lang_name: str) -> Optional[Any]:
  """Dynamically loads modern modular tree-sitter language or legacy tree_sitter_languages."""
  mod_name = f"tree_sitter_{lang_name}"
  try:
    mod = __import__(mod_name)
    if hasattr(mod, f"language_{lang_name}"):
      return tree_sitter.Language(getattr(mod, f"language_{lang_name}")())
    if hasattr(mod, "language"):
      return tree_sitter.Language(mod.language())
  except Exception:
    pass

  try:
    import tree_sitter_languages
    return tree_sitter_languages.get_language(lang_name)
  except Exception:
    return None


TS_TAG_QUERIES = {
  "python": """
    (class_definition name: (identifier) @name.class)
    (function_definition name: (identifier) @name.function)
    (call function: (identifier) @ref.call)
    (call function: (attribute attribute: (identifier) @ref.call))
  """,
  "c": """
    (struct_specifier name: (type_identifier) @name.struct)
    (type_definition declarator: (type_identifier) @name.typedef)
    (function_definition declarator: (function_declarator declarator: (identifier) @name.function))
    (declaration declarator: (function_declarator declarator: (identifier) @name.function_decl))
    (call_expression function: (identifier) @ref.call)
  """,
  "cpp": """
    (class_specifier name: (type_identifier) @name.class)
    (struct_specifier name: (type_identifier) @name.struct)
    (function_definition declarator: (function_declarator declarator: (identifier) @name.function))
    (function_definition declarator: (function_declarator declarator: (field_identifier) @name.method))
    (call_expression function: (identifier) @ref.call)
  """,
  "javascript": """
    (class_declaration name: (identifier) @name.class)
    (function_declaration name: (identifier) @name.function)
    (method_definition name: (property_identifier) @name.method)
    (call_expression function: (identifier) @ref.call)
  """,
  "typescript": """
    (class_declaration name: (identifier) @name.class)
    (interface_declaration name: (identifier) @name.interface)
    (type_alias_declaration name: (type_identifier) @name.type)
    (function_declaration name: (identifier) @name.function)
    (method_definition name: (property_identifier) @name.method)
    (call_expression function: (identifier) @ref.call)
  """,
  "go": """
    (type_spec name: (type_identifier) @name.type)
    (function_declaration name: (identifier) @name.function)
    (method_declaration name: (field_identifier) @name.method)
    (call_expression function: (identifier) @ref.call)
  """,
  "rust": """
    (struct_item name: (type_identifier) @name.struct)
    (enum_item name: (type_identifier) @name.enum)
    (trait_item name: (type_identifier) @name.trait)
    (function_item name: (identifier) @name.function)
    (call_expression function: (identifier) @ref.call)
  """
}

EXT_TO_LANGUAGE = {
  ".py": "python",
  ".c": "c",
  ".h": "c",
  ".cpp": "cpp",
  ".hpp": "cpp",
  ".cc": "cpp",
  ".cxx": "cpp",
  ".js": "javascript",
  ".jsx": "javascript",
  ".ts": "typescript",
  ".tsx": "typescript",
  ".go": "go",
  ".rs": "rust"
}


class RepoMap:
  """Builds a high-density, PageRank-weighted structural repository map."""

  def __init__(self, sandbox_dir: str, max_tokens: int = 1024):
    self.sandbox_dir = os.path.abspath(sandbox_dir)
    self.max_tokens = max_tokens
    self.cache_path = os.path.join(self.sandbox_dir, ".chatty", "cache", "repo_map.json")
    self.cache: Dict[str, Any] = self._load_cache()

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

  def extract_tags(self, rel_path: str, code_str: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Extracts defined symbols and referenced identifiers from source code."""
    ext = os.path.splitext(rel_path)[1].lower()
    lang_name = EXT_TO_LANGUAGE.get(ext)
    defs = []
    refs = set()

    if HAS_TREE_SITTER and lang_name and lang_name in TS_TAG_QUERIES:
      lang = get_tree_sitter_language(lang_name)
      if lang:
        try:
          parser = tree_sitter.Parser(lang)
          tree = parser.parse(code_str.encode("utf-8", errors="ignore"))
          query = tree_sitter.Query(lang, TS_TAG_QUERIES[lang_name])
          cursor = tree_sitter.QueryCursor(query)
          captures = cursor.captures(tree.root_node)

          for tag_name, nodes in captures.items():
            for node in nodes:
              text = node.text.decode("utf-8", errors="ignore")
              if not text:
                continue
              if tag_name.startswith("name."):
                sym_type = tag_name.split(".", 1)[1]
                line = node.start_point[0] + 1
                parent = self._find_parent_name(node)
                defs.append({
                  "name": text,
                  "type": sym_type,
                  "line": line,
                  "parent": parent
                })
              elif tag_name.startswith("ref."):
                refs.add(text)
          return defs, refs
        except Exception:
          pass

    # Python AST fallback
    if ext == ".py":
      try:
        parsed = ast.parse(code_str)
        defs, refs = self._extract_python_ast(parsed)
        return defs, refs
      except Exception:
        pass

    # Regex fallback
    defs, refs = self._extract_regex(code_str)
    return defs, refs

  def _find_parent_name(self, node: Any) -> Optional[str]:
    curr = getattr(node, "parent", None)
    while curr:
      if curr.type in ("class_definition", "class_declaration", "class_specifier", "struct_specifier"):
        for child in curr.children:
          if child.type in ("identifier", "type_identifier"):
            return child.text.decode("utf-8", errors="ignore")
      curr = getattr(curr, "parent", None)
    return None

  def _extract_python_ast(self, parsed: ast.AST) -> Tuple[List[Dict[str, Any]], Set[str]]:
    defs = []
    refs = set()

    class ASTVisitor(ast.NodeVisitor):
      def __init__(self):
        self.current_class = None

      def visit_ClassDef(self, node):
        defs.append({
          "name": node.name,
          "type": "class",
          "line": node.lineno,
          "parent": self.current_class
        })
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

      def visit_FunctionDef(self, node):
        defs.append({
          "name": node.name,
          "type": "function" if not self.current_class else "method",
          "line": node.lineno,
          "parent": self.current_class
        })
        self.generic_visit(node)

      def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

      def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
          refs.add(node.id)
        self.generic_visit(node)

      def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
          refs.add(node.attr)
        self.generic_visit(node)

    visitor = ASTVisitor()
    visitor.visit(parsed)
    return defs, refs

  def _extract_regex(self, code_str: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    defs = []
    refs = set()
    ident_pat = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

    for line_no, line in enumerate(code_str.splitlines(), start=1):
      m = re.match(r"^\s*(def|class|struct|typedef|function|fn)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
      if m:
        defs.append({
          "name": m.group(2),
          "type": m.group(1),
          "line": line_no,
          "parent": None
        })
      for ident in ident_pat.findall(line):
        refs.add(ident)

    return defs, refs

  def scan_files(self) -> Dict[str, Tuple[List[Dict[str, Any]], Set[str]]]:
    """Scans all non-ignored code files in the sandbox, utilizing mtime cache."""
    from chatty.safety import load_ignore_patterns, is_path_ignored
    ignore_patterns = load_ignore_patterns(self.sandbox_dir)

    file_data: Dict[str, Tuple[List[Dict[str, Any]], Set[str]]] = {}
    cache_dirty = False

    for root, dirs, files in os.walk(self.sandbox_dir):
      rel_root = os.path.relpath(root, self.sandbox_dir)
      if rel_root != "." and is_path_ignored(rel_root, ignore_patterns, is_dir=True):
        dirs.clear()
        continue

      for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in EXT_TO_LANGUAGE:
          continue

        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, self.sandbox_dir)
        if is_path_ignored(rel_path, ignore_patterns):
          continue

        try:
          st = os.stat(full_path)
          mtime = st.st_mtime
          # Skip files larger than 500KB
          if st.st_size > 500 * 1024:
            continue
        except OSError:
          continue

        cached_entry = self.cache.get(rel_path)
        if cached_entry and cached_entry.get("mtime") == mtime:
          defs = cached_entry.get("defs", [])
          refs = set(cached_entry.get("refs", []))
          file_data[rel_path] = (defs, refs)
          continue

        try:
          with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
            code_str = fp.read()
          defs, refs = self.extract_tags(rel_path, code_str)
          file_data[rel_path] = (defs, refs)
          self.cache[rel_path] = {
            "mtime": mtime,
            "defs": defs,
            "refs": list(refs)
          }
          cache_dirty = True
        except Exception as e:
          logger.debug(f"Error extracting tags from {rel_path}: {e}")

    if cache_dirty:
      self._save_cache()

    return file_data

  def compute_ranks(self, file_data: Dict[str, Tuple[List[Dict[str, Any]], Set[str]]]) -> Dict[str, float]:
    """Computes PageRank centrality for all files based on definition-reference graph."""
    sym_to_files = defaultdict(set)
    for fname, (defs, _) in file_data.items():
      for d in defs:
        sym_to_files[d["name"]].add(fname)

    graph = defaultdict(set)
    for fname, (_, refs) in file_data.items():
      for r in refs:
        if r in sym_to_files:
          for target in sym_to_files[r]:
            if target != fname:
              graph[fname].add(target)

    all_files = list(file_data.keys())
    N = len(all_files)
    if N == 0:
      return {}

    ranks = {f: 1.0 / N for f in all_files}
    damping = 0.85

    for _ in range(15):
      new_ranks = {f: (1.0 - damping) / N for f in all_files}
      for src, targets in graph.items():
        if not targets:
          continue
        contrib = damping * (ranks[src] / len(targets))
        for t in targets:
          new_ranks[t] += contrib
      ranks = new_ranks

    return ranks

  def generate_map(self) -> str:
    """Generates the formatted repository map string fitting within max_tokens."""
    file_data = self.scan_files()
    if not file_data:
      return ""

    ranks = self.compute_ranks(file_data)
    sorted_files = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    max_chars = self.max_tokens * 4
    blocks = []
    total_chars = 0

    # Count how many files reference each symbol globally
    sym_ref_counts = defaultdict(int)
    for _, (_, refs) in file_data.items():
      for r in refs:
        sym_ref_counts[r] += 1

    for fname, rank in sorted_files:
      defs, _ = file_data[fname]
      if not defs:
        continue

      # Sort defs: first prefer symbols referenced elsewhere, then non-private, then by line
      def sort_key(d):
        name = d["name"]
        ref_count = sym_ref_counts.get(name, 0)
        is_private = name.startswith("_") and not name.startswith("__")
        return (-ref_count, is_private, d.get("line", 0))

      sorted_defs = sorted(defs, key=sort_key)
      # Limit max symbols shown per file to keep breadth
      selected_defs = sorted(sorted_defs[:12], key=lambda d: d.get("line", 0))

      file_lines = [f"{fname}:"]
      for d in selected_defs:
        sym_type = d.get("type", "def")
        name = d.get("name", "")
        parent = d.get("parent")
        if parent:
          file_lines.append(f"  │   {sym_type} {name}")
        else:
          file_lines.append(f"  │ {sym_type} {name}")

      block = "\n".join(file_lines) + "\n"
      if total_chars + len(block) > max_chars:
        break

      blocks.append(block)
      total_chars += len(block)

    return "".join(blocks).strip()
