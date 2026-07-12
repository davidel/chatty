import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from chatty.utils import record_command_binaries, truncate_output

logger = logging.getLogger("chatty")


def cleanup_resources(background_commands: Dict[str, Any]):
  """Kills all active background tasks and removes temporary files."""
  if background_commands:
    if logger is not None:
      try:
        logger.info(f"Cleaning up {len(background_commands)} background commands...")
      except Exception:
        pass
  for task_id, task in list(background_commands.items()):
    proc = task.get("proc")
    status = task.get("status")
    if proc and status is None and proc.poll() is None:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except Exception:
        pass
    try:
      if task.get("stdout_file"):
        task["stdout_file"].close()
      if task.get("stderr_file"):
        task["stderr_file"].close()
    except Exception:
      pass
    try:
      if os is not None and task.get("stdout_path"):
        os.unlink(task["stdout_path"])
    except Exception:
      pass
    try:
      if os is not None and task.get("stderr_path"):
        os.unlink(task["stderr_path"])
    except Exception:
      pass
  background_commands.clear()
  if logger is not None:
    try:
      logger.info("Background commands cleanup finished.")
    except Exception:
      pass


class SubprocessRunner:

  def __init__(self, session: Any):
    self.session = session

  def _apply_output_filters(
    self,
    text: str,
    output_filter: Optional[str] = None,
    head_lines: Optional[int] = None,
    tail_lines: Optional[int] = None
  ) -> str:
    """Applies output filter regex, head/tail limits, and joins lines."""
    if not text:
      return text
    lines = text.splitlines()
    if output_filter:
      try:
        pattern = re.compile(output_filter, re.IGNORECASE)
        lines = [line for line in lines if pattern.search(line)]
      except re.error as e:
        return f"Error applying output_filter: {e}"
    if head_lines is not None and head_lines > 0:
      lines = lines[:head_lines]
    if tail_lines is not None and tail_lines > 0:
      lines = lines[-tail_lines:]
    return "\n".join(lines)

  def _get_pgroup_resources(self, pid: Any) -> Optional[Dict[str, Any]]:
    """Retrieve CPU and RAM usage statistics for the process group of the given pid.

    Returns a dict with CPU percent, RSS RAM bytes, and active process count,
    or None if the process group could not be queried or has exited.
    """
    if pid is None or not isinstance(pid, int):
      return None
    try:
      import psutil
      pgid = os.getpgid(pid)
    except Exception:
      return None

    total_rss = 0
    total_cpu_time = 0.0
    active_processes = 0

    pids_found = []
    for proc in psutil.process_iter(['pid']):
      try:
        p_pid = proc.info['pid']
        if os.getpgid(p_pid) == pgid:
          pids_found.append(p_pid)
          total_rss += proc.memory_info().rss
          cpu_t = proc.cpu_times()
          total_cpu_time += (cpu_t.user + cpu_t.system)
          active_processes += 1
      except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        continue

    if not pids_found:
      return None

    time.sleep(0.1)

    total_cpu_time_2 = 0.0
    for p_pid in pids_found:
      try:
        proc = psutil.Process(p_pid)
        cpu_t = proc.cpu_times()
        total_cpu_time_2 += (cpu_t.user + cpu_t.system)
      except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        continue

    cpu_diff = total_cpu_time_2 - total_cpu_time
    cpu_percent = (cpu_diff / 0.1) * 100.0 if cpu_diff >= 0 else 0.0

    return {
      "cpu_percent": cpu_percent,
      "ram_bytes": total_rss,
      "active_processes": active_processes
    }

  def _format_pgroup_resources(self, pid: Any) -> str:
    """Format the process group resources as a user-friendly string."""
    if pid is None or not isinstance(pid, int):
      return ""
    stats = self.session._get_pgroup_resources(pid)
    if not stats or stats["active_processes"] == 0:
      return ""

    ram_mb = stats["ram_bytes"] / (1024 * 1024)
    if ram_mb >= 1024:
      ram_str = f"{ram_mb / 1024:.2f} GB"
    else:
      ram_str = f"{ram_mb:.1f} MB"

    cpu_str = f"{stats['cpu_percent']:.1f}%"

    return f"Active processes: {stats['active_processes']}, CPU: {cpu_str}, RAM: {ram_str}"

  def tool_run_tests(self, command: str = None) -> str:
    """Run tests in the sandbox, auto-detecting the testing framework if no command is provided."""
    import shutil

    detected_msg = ""
    if not command:
      if os.path.exists(os.path.join(self.session.sandbox, "pytest.ini")) or \
         os.path.exists(os.path.join(self.session.sandbox, "conftest.py")) or \
         os.path.isdir(os.path.join(self.session.sandbox, "tests")) or \
         os.path.isdir(os.path.join(self.session.sandbox, "test")):
        if shutil.which("pytest"):
          command = "pytest"
        else:
          command = "python -m unittest discover"

      elif os.path.exists(os.path.join(self.session.sandbox, "package.json")):
        if shutil.which("npm"):
          command = "npm test"

      elif os.path.exists(os.path.join(self.session.sandbox, "Cargo.toml")):
        if shutil.which("cargo"):
          command = "cargo test"

      elif os.path.exists(os.path.join(self.session.sandbox, "go.mod")):
        if shutil.which("go"):
          command = "go test ./..."

      elif os.path.exists(os.path.join(self.session.sandbox, "CMakeLists.txt")):
        if shutil.which("ctest"):
          command = "ctest"
        elif shutil.which("make"):
          command = "make test"

      elif os.path.exists(os.path.join(self.session.sandbox, "Makefile")) or \
           os.path.exists(os.path.join(self.session.sandbox, "makefile")):
        if shutil.which("make"):
          command = "make test"

      elif os.path.exists(os.path.join(self.session.sandbox, "meson.build")):
        if shutil.which("meson"):
          command = "meson test"
        elif shutil.which("ninja"):
          command = "ninja test"

      if not command:
        return "Error: Could not auto-detect a test suite. Please specify a custom test 'command' (e.g. 'pytest', 'npm test')."

      detected_msg = f"Auto-detected test command: '{command}'\n"

    result = self.tool_run_command(command)
    return detected_msg + result

  def tool_run_command(self, command: str, output_filter: Optional[str] = None, tail_lines: Optional[int] = None, head_lines: Optional[int] = None, combine_stderr: bool = False) -> str:
    """Execute a shell command, transitioning to background execution if it takes too long."""
    logger.info(f"Running shell command: '{command}' (filter={output_filter}, tail={tail_lines}, head={head_lines}, combine_stderr={combine_stderr})")
    validation_err = self.session.validate_command_safety(command)
    if validation_err:
      logger.warning(f"Rejected command '{command}': {validation_err}")
      return validation_err

    task_id = f"task_{self.session.next_task_id}"
    stdout_f = None
    stderr_f = None
    try:
      stdout_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t', prefix=f"chatty_{task_id}_stdout_")
      if not combine_stderr:
        stderr_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t', prefix=f"chatty_{task_id}_stderr_")
      record_command_binaries(command, self.session)

      if self.session.landlock_bin and self.session.sandbox:
        from chatty.landlock import wrap_command_with_landlock
        cmd_args = wrap_command_with_landlock(self.session.landlock_bin, self.session.sandbox, command)
        shell_val = False
      else:
        cmd_args = command
        shell_val = True

      proc = subprocess.Popen(
        cmd_args,
        shell=shell_val,
        cwd=self.session.sandbox,
        stdout=stdout_f,
        stderr=subprocess.STDOUT if combine_stderr else stderr_f,
        start_new_session=True
      )
      try:
        proc.wait(timeout=10)
        stdout_f.close()
        if stderr_f:
          stderr_f.close()
        with open(stdout_f.name, 'r', errors='replace') as f:
          stdout = f.read()
        if stderr_f:
          with open(stderr_f.name, 'r', errors='replace') as f:
            stderr = f.read()
        else:
          stderr = ""
        try:
          os.unlink(stdout_f.name)
        except Exception:
          pass
        if stderr_f:
          try:
            os.unlink(stderr_f.name)
          except Exception:
            pass

        if output_filter or tail_lines is not None or head_lines is not None:
          stdout = self.session._apply_output_filters(stdout, output_filter, head_lines, tail_lines)
          stderr = self.session._apply_output_filters(stderr, output_filter, head_lines, tail_lines)

        output = []
        if stdout:
          output.append(f"Stdout:\n{truncate_output(stdout, max_chars=self.session.max_command_chars)}")
        if stderr:
          output.append(f"Stderr:\n{truncate_output(stderr, max_chars=self.session.max_command_chars)}")
        status = f"Command exited with code {proc.returncode}."
        logger.info(f"Command completed in foreground. Exit code: {proc.returncode}")
        return "\n".join(output) + f"\n{status}" if output else status
      except subprocess.TimeoutExpired:
        task_id = f"task_{self.session.next_task_id}"
        self.session.next_task_id += 1
        logger.info(f"Command timed out. Transitioned to background. Task ID: {task_id}")
        self.session._print(
          f"\n[bold yellow]⚙️  Command took > 10s and is now running in the background. "
          f"Task ID: {task_id}. Use '/status' or check_background_command to monitor progress.[/bold yellow]\n"
        )
        self.session.background_commands[task_id] = {
          "proc": proc,
          "command": command,
          "stdout_path": stdout_f.name,
          "stderr_path": stderr_f.name if stderr_f else None,
          "stdout_file": stdout_f,
          "stderr_file": stderr_f,
          "output_filter": output_filter,
          "tail_lines": tail_lines,
          "head_lines": head_lines
        }
        self.session._prune_background_commands()
        resource_usage = self.session._format_pgroup_resources(getattr(proc, "pid", None))
        resource_msg = f"\nResource usage: {resource_usage}" if resource_usage else ""
        return (
          f"Info: The command is taking longer than 10 seconds. It is now running in the background.\n"
          f"Task ID: {task_id}\n"
          f"You must NOT block. Instead, check its output later by calling the 'check_background_command' tool.{resource_msg}"
        )
    except Exception as e:
      if stdout_f:
        stdout_f.close()
        try:
          os.unlink(stdout_f.name)
        except Exception:
          pass
      if stderr_f:
        stderr_f.close()
        try:
          os.unlink(stderr_f.name)
        except Exception:
          pass
      return f"Error executing command: {str(e)}"

  def tool_check_background_command(
    self,
    task_id: str,
    timeout: Optional[float] = None,
    output_filter: Optional[str] = None,
    tail_lines: Optional[int] = None,
    head_lines: Optional[int] = None
  ) -> str:
    """Check status of a background task and read its currently accumulated stdout and stderr."""
    logger.info(f"Checking status of background task: '{task_id}' (timeout={timeout})")
    task = self.session.background_commands.get(task_id)
    if not task:
      logger.warning(f"Check background task failed: Task ID '{task_id}' not found")
      return f"Error: Task ID '{task_id}' not found."
    proc = task["proc"]

    status = task.get("status")
    timed_out_while_waiting = False
    if status is None:
      if timeout is not None and timeout > 0:
        start_time = time.time()
        while proc.poll() is None:
          elapsed = time.time() - start_time
          if elapsed >= timeout:
            timed_out_while_waiting = True
            break
          time.sleep(min(0.2, timeout - elapsed))
      status = proc.poll()
      if status is not None:
        task["status"] = status

    if status is not None:
      try:
        if task.get("stdout_file"):
          task["stdout_file"].close()
          task["stdout_file"] = None
        if task.get("stderr_file"):
          task["stderr_file"].close()
          task["stderr_file"] = None
      except Exception:
        pass

    try:
      with open(task["stdout_path"], 'r', errors='replace') as f:
        stdout_content = f.read()
      if task.get("stderr_path"):
        with open(task["stderr_path"], 'r', errors='replace') as f:
          stderr_content = f.read()
      else:
        stderr_content = ""
    except Exception as e:
      stdout_content = f"Error reading output: {e}"
      stderr_content = ""

    actual_filter = output_filter if output_filter is not None else task.get("output_filter")
    actual_tail = tail_lines if tail_lines is not None else task.get("tail_lines")
    actual_head = head_lines if head_lines is not None else task.get("head_lines")

    if actual_filter or actual_tail is not None or actual_head is not None:
      stdout_content = self.session._apply_output_filters(stdout_content, actual_filter, actual_head, actual_tail)
      stderr_content = self.session._apply_output_filters(stderr_content, actual_filter, actual_head, actual_tail)

    output = []
    if stdout_content:
      output.append(f"Stdout:\n{truncate_output(stdout_content, max_chars=self.session.max_command_chars)}")
    if stderr_content:
      output.append(f"Stderr:\n{truncate_output(stderr_content, max_chars=self.session.max_command_chars)}")
    if status is None:
      logger.info(f"Task '{task_id}' is STILL RUNNING.")
      status_msg = f"Status: Task '{task_id}' is STILL RUNNING"
      if timeout is not None and timeout > 0 and timed_out_while_waiting:
        status_msg += f" (the check timed out after {timeout} seconds)"
      resource_usage = self.session._format_pgroup_resources(getattr(proc, "pid", None))
      if resource_usage:
        status_msg += f" ({resource_usage})"
      status_msg += ".\n"
      return status_msg + ("\n".join(output) if output else "(No output generated yet)")
    else:
      logger.info(f"Task '{task_id}' FINISHED with exit code {status}.")
      self._prune_background_commands()
      return (
        f"Status: Task '{task_id}' FINISHED with exit code {status}.\n"
        + ("\n".join(output) if output else "(No output generated)")
      )

  def tool_peek_task_output(
    self,
    task_id: str,
    tail_lines: int = 20,
    output_filter: Optional[str] = None
  ) -> str:
    """Peek at the currently accumulated output of a background task without blocking."""
    logger.info(f"Peeking at background task output: '{task_id}' (tail_lines={tail_lines})")
    task = self.session.background_commands.get(task_id)
    if not task:
      logger.warning(f"Peek background task failed: Task ID '{task_id}' not found")
      return f"Error: Task ID '{task_id}' not found."

    proc = task["proc"]
    status = task.get("status")
    if status is None:
      status = proc.poll()
      if status is not None:
        task["status"] = status

    if status is not None:
      try:
        if task.get("stdout_file"):
          task["stdout_file"].close()
          task["stdout_file"] = None
        if task.get("stderr_file"):
          task["stderr_file"].close()
          task["stderr_file"] = None
      except Exception:
        pass

    try:
      with open(task["stdout_path"], 'r', errors='replace') as f:
        stdout_content = f.read()
      if task.get("stderr_path"):
        with open(task["stderr_path"], 'r', errors='replace') as f:
          stderr_content = f.read()
      else:
        stderr_content = ""
    except Exception as e:
      stdout_content = f"Error reading output: {e}"
      stderr_content = ""

    if output_filter or tail_lines is not None:
      stdout_content = self.session._apply_output_filters(stdout_content, output_filter, None, tail_lines)
      stderr_content = self.session._apply_output_filters(stderr_content, output_filter, None, tail_lines)

    output = []
    if stdout_content:
      output.append(f"Stdout:\n{truncate_output(stdout_content, max_chars=self.session.max_command_chars)}")
    if stderr_content:
      output.append(f"Stderr:\n{truncate_output(stderr_content, max_chars=self.session.max_command_chars)}")

    if status is None:
      status_msg = f"Status: Task '{task_id}' is STILL RUNNING"
      resource_usage = self.session._format_pgroup_resources(getattr(proc, "pid", None))
      if resource_usage:
        status_msg += f" ({resource_usage})"
    else:
      status_msg = f"Status: Task '{task_id}' FINISHED with exit code {status}"
      self.session._prune_background_commands()

    peek_output = "\n".join(output) if output else "(No output generated yet)"
    return f"{status_msg}.\n{peek_output}"

  def tool_kill_process(self, task_id: str) -> str:
    """Terminate a background task/process by its Task ID."""
    logger.info(f"Terminating background task: '{task_id}'")
    task = self.session.background_commands.get(task_id)
    if not task:
      logger.warning(f"Kill process failed: Task ID '{task_id}' not found")
      return f"Error: Task ID '{task_id}' not found."

    proc = task["proc"]
    status = task.get("status")
    if status is None:
      status = proc.poll()
      if status is not None:
        task["status"] = status
    if status is None:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
        logger.info(f"Process group {proc.pid} terminated.")
      except Exception as e:
        logger.error(f"Failed to kill process group {proc.pid}: {e}")
        return f"Error terminating process: {e}"
      message = f"Successfully terminated background task '{task_id}'."
    else:
      message = f"Background task '{task_id}' had already exited with code {status}. Cleaned up resources."

    try:
      if task.get("stdout_file"):
        task["stdout_file"].close()
      if task.get("stderr_file"):
        task["stderr_file"].close()
    except Exception:
      pass
    try:
      os.unlink(task["stdout_path"])
      if task.get("stderr_path"):
        os.unlink(task["stderr_path"])
    except Exception:
      pass

    del self.session.background_commands[task_id]
    return message

  def cleanup_background_commands(self):
    """Kills all active background tasks and removes temporary files."""
    cleanup_resources(self.session.background_commands)

  def _prune_background_commands(self):
    """Ensures we only keep the latest max_completed_tasks completed background task outputs, unlinking older ones."""
    completed_tasks = []
    for task_id, task in self.session.background_commands.items():
      status = task.get("status")
      if status is None:
        proc = task.get("proc")
        if proc:
          status = proc.poll()
          if status is not None:
            task["status"] = status
      if status is not None:
        completed_tasks.append(task_id)

    def get_task_num(tid):
      try:
        return int(tid.split("_")[1])
      except (IndexError, ValueError):
        return 0

    completed_tasks.sort(key=get_task_num)
    if len(completed_tasks) > self.session.max_completed_tasks:
      tasks_to_prune = completed_tasks[:-self.session.max_completed_tasks]
      for task_id in tasks_to_prune:
        task = self.session.background_commands[task_id]
        logger.info(f"Pruning old completed background task: '{task_id}'")
        try:
          if task.get("stdout_file"):
            task["stdout_file"].close()
          if task.get("stderr_file"):
            task["stderr_file"].close()
        except Exception:
          pass
        try:
          if task.get("stdout_path"):
            os.unlink(task["stdout_path"])
          if task.get("stderr_path"):
            os.unlink(task["stderr_path"])
        except Exception:
          pass
        del self.session.background_commands[task_id]
