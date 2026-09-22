import os
import shutil
import tempfile
import time
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.repo_map import RepoMap
from chatty.session import ChatbotSession
from chatty.commands import cmd_repo_map


class TestRepoMap(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = tempfile.mkdtemp()
    self.repo_map = RepoMap(self.temp_dir, max_tokens=500)

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.temp_dir)

  def test_python_tag_extraction(self):
    code = """
class Worker:
  def do_work(self):
    return 42

def run():
  w = Worker()
  w.do_work()
"""
    defs, refs = self.repo_map.extract_tags("worker.py", code)
    names = [d["name"] for d in defs]
    self.assertIn("Worker", names)
    self.assertIn("do_work", names)
    self.assertIn("run", names)

    # References should include Worker and do_work
    self.assertIn("Worker", refs)
    self.assertIn("do_work", refs)

  def test_c_tag_extraction(self):
    c_code = """
struct Point {
  int x;
  int y;
};

int compute(int a) {
  return a * 2;
}

int main() {
  return compute(10);
}
"""
    defs, refs = self.repo_map.extract_tags("main.c", c_code)
    names = [d["name"] for d in defs]
    self.assertIn("compute", names)
    self.assertIn("main", names)
    self.assertIn("compute", refs)

  def test_cpp_tag_extraction(self):
    cpp_code = """
class Engine {
public:
  void start();
};

void run_engine() {
  Engine e;
  e.start();
}
"""
    defs, refs = self.repo_map.extract_tags("engine.cpp", cpp_code)
    names = [d["name"] for d in defs]
    self.assertIn("Engine", names)
    self.assertIn("run_engine", names)

  def test_pagerank_ranking(self):
    # File A has a utility function
    with open(os.path.join(self.temp_dir, "utils.py"), "w", encoding="utf-8") as f:
      f.write("def helper():\n  return 1\n")

    # File B and C both call helper() from utils.py
    with open(os.path.join(self.temp_dir, "app.py"), "w", encoding="utf-8") as f:
      f.write("from utils import helper\ndef main():\n  helper()\n")

    with open(os.path.join(self.temp_dir, "service.py"), "w", encoding="utf-8") as f:
      f.write("from utils import helper\ndef process():\n  helper()\n")

    file_data = self.repo_map.scan_files()
    ranks = self.repo_map.compute_ranks(file_data)

    # utils.py is referenced by app.py and service.py, so it should have highest rank
    self.assertIn("utils.py", ranks)
    self.assertIn("app.py", ranks)
    self.assertIn("service.py", ranks)
    self.assertGreater(ranks["utils.py"], ranks["app.py"])
    self.assertGreater(ranks["utils.py"], ranks["service.py"])

  def test_token_budget_truncation(self):
    # Create several files with multiple symbols
    for i in range(10):
      with open(os.path.join(self.temp_dir, f"module_{i}.py"), "w", encoding="utf-8") as f:
        funcs = "\n".join([f"def func_{i}_{j}():\n  pass\n" for j in range(10)])
        f.write(funcs)

    # Very small budget (e.g. 20 tokens -> ~80 characters)
    small_map = RepoMap(self.temp_dir, max_tokens=20)
    result = small_map.generate_map()
    self.assertTrue(len(result) <= 20 * 4 + 20)

  def test_caching_and_invalidation(self):
    file_path = os.path.join(self.temp_dir, "cached.py")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write("def original_function():\n  pass\n")

    # First scan populates cache
    self.repo_map.scan_files()
    cache_file = os.path.join(self.temp_dir, ".chatty", "cache", "repo_map.json")
    self.assertTrue(os.path.exists(cache_file))

    with open(cache_file, "r", encoding="utf-8") as f:
      import json
      cache_data = json.load(f)
    self.assertIn("cached.py", cache_data)
    orig_mtime = cache_data["cached.py"]["mtime"]

    # Sleep slightly to ensure mtime changes
    time.sleep(0.05)
    with open(file_path, "w", encoding="utf-8") as f:
      f.write("def modified_function():\n  pass\n")

    new_file_data = self.repo_map.scan_files()
    defs, _ = new_file_data["cached.py"]
    names = [d["name"] for d in defs]
    self.assertIn("modified_function", names)
    self.assertNotIn("original_function", names)

  def test_session_repo_map_integration(self):
    with open(os.path.join(self.temp_dir, "main.py"), "w", encoding="utf-8") as f:
      f.write("class AppRunner:\n  def start(self):\n    pass\n")

    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.temp_dir,
      repo_map=True,
      repo_map_tokens=500
    )

    repo_map_content = session.get_repo_map()
    self.assertIn("main.py:", repo_map_content)
    self.assertIn("AppRunner", repo_map_content)

    system_prompt = session.get_active_system_prompt()
    self.assertIn("## Repository Map", system_prompt)
    self.assertIn("AppRunner", system_prompt)

    # Disable repo map
    session.config.repo_map = False
    system_prompt_disabled = session.get_active_system_prompt()
    self.assertNotIn("Repository Structure Map", system_prompt_disabled)

  def test_cmd_repo_map(self):
    with open(os.path.join(self.temp_dir, "core.py"), "w", encoding="utf-8") as f:
      f.write("def core_logic():\n  pass\n")

    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.temp_dir,
      headless=True
    )

    res = cmd_repo_map(session, "")
    self.assertTrue(res)

    res_refresh = cmd_repo_map(session, "refresh")
    self.assertTrue(res_refresh)


if __name__ == "__main__":
  unittest.main()
