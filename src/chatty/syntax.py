import ast
import json
import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple
import yaml
from chatty.utils import record_command_binaries


def validate_file_syntax(path: str, content: str, sandbox_dir: str = None, compile_paths: List[str] = None) -> Tuple[bool, str]:
  """
  Checks syntax of content based on file extension.
  Returns (True, "") if syntax is valid or language has no validator.
  Returns (False, "Error message") if syntax verification fails.
  """
  ext = os.path.splitext(path)[1].lower()
  
  if ext == ".py":
    try:
      ast.parse(content)
    except SyntaxError as e:
      return False, f"Syntax Error: {e.msg} on line {e.lineno}, column {e.offset}\nLine content: {e.text}"
      
  elif ext == ".json":
    try:
      json.loads(content)
    except json.JSONDecodeError as e:
      return False, f"JSON Parsing Error: {e.msg} at line {e.lineno}, column {e.colno}"
      
  elif ext in (".yaml", ".yml"):
    try:
      yaml.safe_load(content)
    except Exception as e:
      return False, f"YAML Parsing Error: {str(e)}"
      
  elif ext in (".c", ".cpp", ".h", ".hpp"):
    try:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w+t') as temp:
        temp.write(content)
        temp_name = temp.name
      try:
        is_cpp = ext in (".cpp", ".hpp")
        compiler = "clang++" if shutil.which("clang++") else "g++" if is_cpp else "clang" if shutil.which("clang") else "gcc"
          
        cmd_args = [compiler, "-fsyntax-only"]
        target_dir = os.path.dirname(path) or "."
        cmd_args.extend(["-I", target_dir])
        if sandbox_dir:
          for p in (compile_paths or []):
            abs_p = os.path.abspath(os.path.join(sandbox_dir, p))
            if os.path.isdir(abs_p):
              cmd_args.extend(["-I", abs_p])
            elif os.path.isfile(abs_p):
              cmd_args.extend(["-I", os.path.dirname(abs_p)])
        cmd_args.append(temp_name)
          
        record_command_binaries(cmd_args)
        proc = subprocess.run(
          cmd_args,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          timeout=3
        )
        if proc.returncode != 0:
          err_msg = proc.stderr.replace(temp_name, os.path.basename(path))
          return False, f"C/C++ Compiler Error:\n{err_msg}"
      finally:
        try:
          os.unlink(temp_name)
        except Exception:
          pass
    except Exception:
      pass

  elif ext in (".v", ".sv", ".vh", ".svh"):
    try:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w+t') as temp:
        temp.write(content)
        temp_name = temp.name
      try:
        dir_name = os.path.dirname(path) or "."
        search_dirs = {dir_name}
        extra_files = []
        if sandbox_dir:
          try:
            for root, dirs, files in os.walk(sandbox_dir):
              if any(f.endswith((".v", ".sv", ".vh", ".svh")) for f in files):
                search_dirs.add(root)
          except Exception:
            pass

          for p in (compile_paths or []):
            abs_p = os.path.abspath(os.path.join(sandbox_dir, p))
            if os.path.isdir(abs_p):
              search_dirs.add(abs_p)
            elif os.path.isfile(abs_p):
              extra_files.append(abs_p)
              search_dirs.add(os.path.dirname(abs_p))

        if shutil.which("verilator"):
          cmd_args = ["verilator", "--lint-only", "-Wno-fatal", "-Wno-MODMISSING"]
          for s_dir in sorted(search_dirs):
            cmd_args.extend(["-y", s_dir, f"-I{s_dir}"])
          cmd_args.extend(extra_files)
          cmd_args.append(temp_name)
          record_command_binaries(cmd_args)
          proc = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, (
              f"Verilator Lint Error:\n{err_msg}\n"
              "Note: If Verilator cannot resolve dependencies, make sure the required module or include files are present. "
            )
        elif shutil.which("iverilog"):
          cmd_args = ["iverilog", "-g2012", "-t", "null"]
          for s_dir in sorted(search_dirs):
            cmd_args.extend(["-y", s_dir, f"-I{s_dir}"])
          cmd_args.extend(extra_files)
          cmd_args.append(temp_name)
          record_command_binaries(cmd_args)
          proc = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, (
              f"Icarus Verilog Syntax Error:\n{err_msg}\n"
            )
      finally:
        try:
          os.unlink(temp_name)
        except Exception:
          pass
    except Exception:
      pass

  elif ext in (".vhd", ".vhdl"):
    try:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w+t') as temp:
        temp.write(content)
        temp_name = temp.name
      try:
        if shutil.which("ghdl"):
          cmd_args = ["ghdl", "-s"]
          if sandbox_dir:
            for p in (compile_paths or []):
              abs_p = os.path.abspath(os.path.join(sandbox_dir, p))
              if os.path.isdir(abs_p):
                cmd_args.append(f"-P{abs_p}")
              elif os.path.isfile(abs_p):
                cmd_args.append(f"-P{os.path.dirname(abs_p)}")
          cmd_args.append(temp_name)
          record_command_binaries(cmd_args)
          proc = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, f"GHDL Syntax Error:\n{err_msg}"
      finally:
        try:
          os.unlink(temp_name)
        except Exception:
          pass
    except Exception:
      pass
      
  return True, ""
