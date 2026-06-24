import datetime
import logging


class GlogFormatter(logging.Formatter):
  """A logging formatter that formats messages like Google's glog,
  but using process ID instead of thread ID.
  Format: Lyyyymmdd hh:mm:ss.uuuuuu process file:line] msg
  """

  def format(self, record):
    level_char = "I"
    if record.levelname:
      if record.levelname == "CRITICAL":
        level_char = "F"
      elif record.levelname in ("DEBUG", "INFO", "WARNING", "ERROR"):
        level_char = record.levelname[0]
      else:
        level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    time_str = dt.strftime("%Y%m%d %H:%M:%S.%f")
    pid = record.process
    filename = record.filename
    lineno = record.lineno
    record.message = record.getMessage()
    prefix = f"{level_char}{time_str} {pid} {filename}:{lineno}]"
    s = f"{prefix} {record.message}"
    if record.exc_info:
      if not record.exc_text:
        record.exc_text = self.formatException(record.exc_info)
    if record.exc_text:
      if s[-1:] != "\n":
        s = s + "\n"
      s = s + record.exc_text
    if record.stack_info:
      if s[-1:] != "\n":
        s = s + "\n"
      s = s + self.formatStack(record.stack_info)
    return s


def setup_logging(log_file: str, log_level_str: str) -> None:
  """Configure logging with the custom GlogFormatter."""
  log_level = getattr(logging, log_level_str.upper(), logging.INFO)
  handler = logging.FileHandler(log_file, encoding="utf-8")
  handler.setFormatter(GlogFormatter())
  logging.basicConfig(
    level=log_level,
    handlers=[handler]
  )
