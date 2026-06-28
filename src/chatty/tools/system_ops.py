import time
import json
from typing import List


def tool_sleep(seconds: float) -> str:
  """Sleep for a specified number of seconds to wait for background operations to progress."""
  try:
    sec = float(seconds)
    if sec < 0:
      return "Error: sleep duration cannot be negative."
    if sec > 60:
      return "Error: maximum sleep duration is 60 seconds."
    time.sleep(sec)
    return f"Successfully slept for {sec} seconds."
  except Exception as e:
    return f"Error sleeping: {str(e)}"


def tool_ask_question(question: str, options: List[str] = None, multiple: bool = False) -> str:
  """Prompt the user with a question and optional list of selections, returning their answer."""
  from rich.console import Console
  from rich.panel import Panel
  from prompt_toolkit import prompt
  
  c = Console()
  c.print()
  
  if options:
    choices_text = ""
    for idx, opt in enumerate(options, 1):
      choices_text += f"[bold cyan]{idx}.[/bold cyan] {opt}\n"
    
    c.print(Panel(
      f"[bold]{question}[/bold]\n\n{choices_text.strip()}",
      title="❓ Question for User",
      border_style="magenta",
      expand=False
    ))
    
    prompt_msg = "Select option number(s) (comma separated)" if multiple else "Select option number or type custom response"
    while True:
      try:
        ans = prompt(f"{prompt_msg} > ")
        ans_strip = ans.strip()
        if not ans_strip:
          continue
        
        if multiple:
          parts = [p.strip() for p in ans_strip.split(",")]
          selected = []
          invalid = False
          for p in parts:
            if p.isdigit():
              idx = int(p)
              if 1 <= idx <= len(options):
                selected.append(options[idx - 1])
              else:
                invalid = True
                break
            else:
              invalid = True
              break
          if not invalid and selected:
            return json.dumps({"selection": selected})
          else:
            return json.dumps({"custom_response": ans_strip})
        else:
          if ans_strip.isdigit():
            idx = int(ans_strip)
            if 1 <= idx <= len(options):
              return json.dumps({"selection": options[idx - 1]})
          return json.dumps({"custom_response": ans_strip})
      except (KeyboardInterrupt, EOFError):
        return json.dumps({"error": "User cancelled the prompt."})
  else:
    c.print(Panel(
      f"[bold]{question}[/bold]",
      title="❓ Question for User",
      border_style="magenta",
      expand=False
    ))
    try:
      ans = prompt("Answer > ")
      return json.dumps({"response": ans.strip()})
    except (KeyboardInterrupt, EOFError):
      return json.dumps({"error": "User cancelled the prompt."})
