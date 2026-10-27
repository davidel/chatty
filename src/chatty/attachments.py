import base64
import io
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


SUPPORTED_EXTENSIONS = {
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".bmp",
  ".svg",
}


INLINE_PATH_PATTERN = re.compile(
  r'(?:^|[\s(\[{<,;])@(?:'
  r'\"(?P<qpath>[^"]+)\"|'
  r"\'(?P<sqpath>[^']+)\'|"
  r'(?P<rawpath>[^\s()\[\]{}<>\"\'!?,;:]+)'
  r')'
  r'(?::(?P<spec>[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*|all))?'
  r'(?=[.,;:?!)}\]]*(?:\s|$))'
)


@dataclass
class StagedAttachment:
  original_path: str
  filename: str
  staged_path: str
  media_type: str
  page_numbers: List[int] = field(default_factory=list)
  images: List[Tuple[int, bytes]] = field(default_factory=list)
  extracted_text: Optional[str] = None
  warning: Optional[str] = None


def parse_page_spec(spec_str: Optional[str], total_pages: Optional[int] = None) -> List[int]:
  """
  Parses page specifications like '1', '1-3', '1,3,5', or 'all'.
  Defaults to pages 1-3 (or total_pages if fewer) when spec_str is omitted.
  """
  if not spec_str or not spec_str.strip():
    if total_pages:
      return list(range(1, min(total_pages, 3) + 1))
    return [1, 2, 3]

  cleaned = spec_str.strip().lower()
  if cleaned == "all":
    if total_pages:
      return list(range(1, total_pages + 1))
    return [1, 2, 3]

  pages = set()
  parts = [p.strip() for p in cleaned.split(",") if p.strip()]
  for part in parts:
    if "-" in part:
      sub_parts = part.split("-", 1)
      try:
        start_p = int(sub_parts[0].strip())
        end_p = int(sub_parts[1].strip())
        if start_p <= end_p:
          for p in range(start_p, end_p + 1):
            if p >= 1 and (total_pages is None or p <= total_pages):
              pages.add(p)
      except ValueError:
        pass
    else:
      try:
        p = int(part)
        if p >= 1 and (total_pages is None or p <= total_pages):
          pages.add(p)
      except ValueError:
        pass

  result = sorted(pages)
  if not result:
    if total_pages:
      return list(range(1, min(total_pages, 3) + 1))
    return [1]
  return result


def detect_pdf_engine() -> Tuple[Optional[str], Optional[str]]:
  """
  Detects available PDF rendering engines in order of preference:
  1. pypdfium2 (self-contained wheel)
  2. pdf2image (requires pdftoppm)
  3. fitz (PyMuPDF)
  Returns (engine_name, None) if found, or (None, recommendation) if missing.
  """
  try:
    import pypdfium2
    return "pypdfium2", None
  except ImportError:
    pass

  try:
    import pdf2image
    if shutil.which("pdftoppm"):
      return "pdf2image", None
  except ImportError:
    pass

  try:
    import fitz
    return "fitz", None
  except ImportError:
    pass

  recommendation = (
    "Visual PDF rendering requires a PDF-to-image engine.\n"
    "To enable visual schematic ingestion, please install either:\n"
    "  • pip install pypdfium2 (recommended, self-contained)\n"
    "  • sudo apt install poppler-utils && pip install pdf2image\n"
    "  • pip install pymupdf"
  )
  return None, recommendation


def get_pdf_total_pages(pdf_path: str) -> int:
  """Returns the total number of pages in a PDF file using pypdf if available."""
  try:
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    return len(reader.pages)
  except Exception:
    return 1


def rasterize_pdf_pages(
  pdf_path: str,
  page_numbers: List[int],
  dpi: int = 150
) -> List[Tuple[int, bytes]]:
  """
  Renders specified 1-indexed pages of a PDF to PNG byte buffers.
  Returns list of (page_number, png_bytes).
  """
  engine, rec = detect_pdf_engine()
  if not engine:
    raise RuntimeError(rec)

  results: List[Tuple[int, bytes]] = []

  if engine == "pypdfium2":
    import pypdfium2
    pdf = pypdfium2.PdfDocument(pdf_path)
    scale = dpi / 72.0
    for p_num in page_numbers:
      idx = p_num - 1
      if 0 <= idx < len(pdf):
        page = pdf[idx]
        pil_image = page.render(scale=scale).to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        results.append((p_num, buf.getvalue()))

  elif engine == "pdf2image":
    import pdf2image
    for p_num in page_numbers:
      images = pdf2image.convert_from_path(
        pdf_path,
        first_page=p_num,
        last_page=p_num,
        dpi=dpi
      )
      if images:
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        results.append((p_num, buf.getvalue()))

  elif engine == "fitz":
    import fitz
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for p_num in page_numbers:
      idx = p_num - 1
      if 0 <= idx < len(doc):
        page = doc[idx]
        pix = page.get_pixmap(matrix=mat)
        results.append((p_num, pix.tobytes("png")))

  return results


def get_attachments_dir(sandbox_dir: str) -> str:
  """Returns the path to .chatty/attachments/ directory inside sandbox."""
  attach_dir = os.path.join(sandbox_dir, ".chatty", "attachments")
  os.makedirs(attach_dir, exist_ok=True)
  return attach_dir


def stage_attachment(
  sandbox_dir: str,
  file_path: str,
  pages_spec: Optional[str] = None
) -> Tuple[Optional[StagedAttachment], Optional[str]]:
  """
  Stages a local file (PDF, image, SVG) for prompt attachment.
  Returns (StagedAttachment, None) on success or (None, error_message) on failure.
  """
  abs_path = os.path.abspath(file_path)
  if not os.path.exists(abs_path):
    return None, f"File not found: {file_path}"
  if not os.path.isfile(abs_path):
    return None, f"Path is not a file: {file_path}"

  filename = os.path.basename(abs_path)
  ext = os.path.splitext(filename)[1].lower()
  if ext not in SUPPORTED_EXTENSIONS:
    return None, f"Unsupported attachment format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

  attachments_dir = get_attachments_dir(sandbox_dir)
  unique_prefix = uuid.uuid4().hex[:8]
  staged_filename = f"{unique_prefix}_{filename}"
  staged_dest = os.path.join(attachments_dir, staged_filename)
  shutil.copy2(abs_path, staged_dest)

  warning_msg: Optional[str] = None

  if ext == ".pdf":
    total_pages = get_pdf_total_pages(abs_path)
    selected_pages = parse_page_spec(pages_spec, total_pages)

    extracted_text = ""
    try:
      import pypdf
      reader = pypdf.PdfReader(abs_path)
      text_parts = []
      for p_num in selected_pages:
        idx = p_num - 1
        if 0 <= idx < len(reader.pages):
          txt = reader.pages[idx].extract_text()
          if txt and txt.strip():
            text_parts.append(f"--- Page {p_num} ---\n{txt.strip()}")
      if text_parts:
        extracted_text = "\n\n".join(text_parts)
    except Exception:
      pass

    rendered_images = []
    try:
      rendered_images = rasterize_pdf_pages(abs_path, selected_pages)
    except RuntimeError as re_err:
      warning_msg = str(re_err)

    attachment = StagedAttachment(
      original_path=abs_path,
      filename=filename,
      staged_path=staged_dest,
      media_type="pdf",
      page_numbers=selected_pages,
      images=rendered_images,
      extracted_text=extracted_text if extracted_text else None,
      warning=warning_msg
    )
    return attachment, None

  elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
    try:
      with open(abs_path, "rb") as f:
        img_bytes = f.read()
      attachment = StagedAttachment(
        original_path=abs_path,
        filename=filename,
        staged_path=staged_dest,
        media_type="image",
        page_numbers=[1],
        images=[(1, img_bytes)],
        warning=None
      )
      return attachment, None
    except Exception as e:
      return None, f"Failed to read image '{filename}': {e}"

  elif ext == ".svg":
    svg_text = ""
    try:
      with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        svg_text = f.read()
    except Exception:
      pass

    rendered_images = []
    try:
      import cairosvg
      png_bytes = cairosvg.svg2png(url=abs_path)
      if png_bytes:
        rendered_images.append((1, png_bytes))
    except Exception:
      pass

    attachment = StagedAttachment(
      original_path=abs_path,
      filename=filename,
      staged_path=staged_dest,
      media_type="svg",
      page_numbers=[1],
      images=rendered_images,
      extracted_text=svg_text if svg_text else None,
      warning=None
    )
    return attachment, None

  return None, f"Unhandled attachment format '{ext}'"


def format_multimodal_user_message(
  user_text: str,
  attachments: List[StagedAttachment]
) -> Dict[str, Any]:
  """
  Builds a standard OpenAI/LiteLLM multimodal user message dict.
  Incorporates text instructions, extracted document annotations, and base64 image data URLs.
  """
  if not attachments:
    return {"role": "user", "content": user_text}

  has_images = any(len(att.images) > 0 for att in attachments)
  has_text = any(att.extracted_text for att in attachments)

  if not has_images and not has_text:
    return {"role": "user", "content": user_text}

  content_blocks: List[Dict[str, Any]] = []

  if user_text and user_text.strip():
    content_blocks.append({"type": "text", "text": user_text})

  for att in attachments:
    if att.extracted_text:
      header = f"\n\n--- Attachment: {att.filename} ---"
      content_blocks.append({
        "type": "text",
        "text": f"{header}\n{att.extracted_text}\n--- End of {att.filename} ---"
      })

    for p_num, img_bytes in att.images:
      b64_data = base64.b64encode(img_bytes).decode("ascii")
      content_blocks.append({
        "type": "image_url",
        "image_url": {
          "url": f"data:image/png;base64,{b64_data}",
          "detail": "high"
        }
      })

  return {
    "role": "user",
    "content": content_blocks
  }


def extract_inline_paths(text: str) -> List[Tuple[str, Optional[str]]]:
  """
  Extracts inline @path mentions from text.
  Returns a list of (path_str, optional_pages_spec) tuples.
  """
  if not text or "@" not in text:
    return []

  results: List[Tuple[str, Optional[str]]] = []
  for match in INLINE_PATH_PATTERN.finditer(text):
    is_quoted = bool(match.group("qpath") or match.group("sqpath"))
    path = match.group("qpath") or match.group("sqpath") or match.group("rawpath")
    spec = match.group("spec")
    if path and path.strip():
      clean_path = path.strip() if is_quoted else path.strip().rstrip(".,;:?!")
      if clean_path:
        results.append((clean_path, spec.strip() if spec else None))

  return results


def resolve_inline_path(path_str: str, sandbox_dir: Optional[str] = None) -> Optional[str]:
  """
  Resolves an inline path reference to an existing file path, checking:
  1. Absolute / tilde-expanded paths
  2. Relative to sandbox directory (if provided)
  3. Relative to current working directory
  Returns absolute path if file exists, else None.
  """
  expanded = os.path.expanduser(path_str.strip())
  if os.path.isabs(expanded):
    if os.path.isfile(expanded):
      return os.path.abspath(expanded)
    return None

  if sandbox_dir:
    candidate = os.path.join(sandbox_dir, expanded)
    if os.path.isfile(candidate):
      return os.path.abspath(candidate)

  cwd_candidate = os.path.abspath(expanded)
  if os.path.isfile(cwd_candidate):
    return cwd_candidate

  return None


def stage_text_file(
  sandbox_dir: str,
  file_path: str,
  max_bytes: int = 100000
) -> Tuple[Optional[StagedAttachment], Optional[str]]:
  """
  Stages a local text or code file as context for prompt attachments.
  """
  abs_path = os.path.abspath(file_path)
  if not os.path.exists(abs_path):
    return None, f"File not found: {file_path}"
  if not os.path.isfile(abs_path):
    return None, f"Path is not a file: {file_path}"

  filename = os.path.basename(abs_path)
  try:
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
      content = f.read(max_bytes + 1)
    if len(content) > max_bytes:
      content = content[:max_bytes] + f"\n\n... (truncated at {max_bytes} bytes) ..."

    attachments_dir = get_attachments_dir(sandbox_dir)
    staged_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
    staged_dest = os.path.join(attachments_dir, staged_filename)
    try:
      shutil.copy2(abs_path, staged_dest)
    except Exception:
      staged_dest = abs_path

    attachment = StagedAttachment(
      original_path=abs_path,
      filename=filename,
      staged_path=staged_dest,
      media_type="text",
      page_numbers=[1],
      images=[],
      extracted_text=content,
      warning=None
    )
    return attachment, None
  except Exception as e:
    return None, f"Failed to stage text file '{filename}': {e}"


def process_inline_attachments(
  user_input: str,
  sandbox_dir: Optional[str] = None
) -> Tuple[List[StagedAttachment], List[str]]:
  """
  Finds @path mentions in user_input, resolves existing files, and stages them.
  Returns (attachments, warnings).
  """
  extracted = extract_inline_paths(user_input)
  if not extracted:
    return [], []

  target_sandbox = sandbox_dir or os.getcwd()
  attachments: List[StagedAttachment] = []
  warnings: List[str] = []
  seen_paths = set()

  for path_str, spec in extracted:
    resolved = resolve_inline_path(path_str, target_sandbox)
    if not resolved:
      continue

    cache_key = (resolved, spec)
    if cache_key in seen_paths:
      continue
    seen_paths.add(cache_key)

    ext = os.path.splitext(resolved)[1].lower()
    if ext in SUPPORTED_EXTENSIONS:
      att, err = stage_attachment(target_sandbox, resolved, pages_spec=spec)
    else:
      att, err = stage_text_file(target_sandbox, resolved)

    if att:
      attachments.append(att)
      if att.warning:
        warnings.append(att.warning)
    elif err:
      warnings.append(err)

  return attachments, warnings
