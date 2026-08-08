"""Module Caption: sinh caption (free-text + structured JSON) cho Frame/Clip/Shot bằng VLM.

Xem ``Markdown_Doc/module_caption.md`` (đặc tả) và
``Markdown_Doc/caption_pipeline.md`` (chi tiết pipeline, logic cài đặt, và
các vấn đề thực tế gặp phải) để có đầy đủ ngữ cảnh.
"""

from BackEnd.app.caption.caption_module import CaptionGenerator

__all__ = ["CaptionGenerator"]
