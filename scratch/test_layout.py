import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.segment import Segment

class SegmentsRenderable:
  def __init__(self, lines):
    self.lines = lines
  def __rich_console__(self, console, options):
    for line in self.lines:
      yield from line

class LiveScreenLayout:
  def __init__(self, panels, status_bar, console):
    self.panels = panels
    self.status_bar = status_bar
    self.console = console

  def __rich_console__(self, console, options):
    W = console.width
    H = console.height
    
    status_bar_height = 1 if self.status_bar is not None else 0
    max_panels_height = max(1, H - status_bar_height - 1)
    
    temp_console = Console(width=max(4, W - 4))
    
    panel_lines = []
    full_heights = []
    for p in self.panels:
      content = p["content"]
      if isinstance(content, str):
        content = Markdown(content)
      lines = temp_console.render_lines(content, new_lines=True)
      panel_lines.append(lines)
      full_heights.append(len(lines) + 2)
      
    heights = self.allocate_heights(full_heights, max_panels_height)
    
    rendered_panels = []
    total_panels_height = 0
    for i, p in enumerate(self.panels):
      h = heights[i]
      if h >= 3:
        inner_height = h - 2
        scrolled = panel_lines[i][-inner_height:]
        rendered_panel = Panel(
          SegmentsRenderable(scrolled),
          title=p["title"],
          border_style=p["border_style"],
          height=h
        )
        rendered_panels.append(rendered_panel)
        total_panels_height += h
        
    H_target = H - 1
    padding_height = H_target - total_panels_height - status_bar_height
    
    if padding_height > 0:
      yield Segment("\n" * padding_height)
      
    for rp in rendered_panels:
      yield from console.render(rp, options)
      
    if self.status_bar is not None:
      yield from console.render(self.status_bar, options)

  def allocate_heights(self, full_heights, max_height):
    n = len(full_heights)
    if n == 0:
      return []
    heights = [0] * n
    heights[-1] = min(full_heights[-1], max_height)
    if heights[-1] < 3 and full_heights[-1] >= 3:
      heights[-1] = min(3, max_height)
      
    remaining = max_height - heights[-1]
    for i in range(n - 2, -1, -1):
      if remaining >= 3:
        allocated = min(full_heights[i], remaining)
        if allocated < 3:
          allocated = 0
        heights[i] = allocated
        remaining -= allocated
      else:
        heights[i] = 0
    return heights

console = Console(width=40, height=15)
panels = [
  {"title": "Thinking", "content": "I am thinking about a solution.\n\nIt has multiple steps.\n\nFirst step is parsing.\n\nSecond step is compiling.\n\nThird step is running.", "border_style": "yellow"},
  # Very long assistant response
  {"title": "Assistant", "content": "Hello! I am a chatbot.\n\nI can write python codes for you.\n\nLet me show you how to do this.\n\nAnother line of response.\n\nAnd another one.\n\nAnd another one.\n\nAnd another one.\n\nAnd another one.", "border_style": "green"}
]
status_bar = Panel("Status Bar", height=1)

layout = LiveScreenLayout(panels, status_bar, console)
console.print(layout)
