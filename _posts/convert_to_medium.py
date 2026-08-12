import os
import re
import urllib.parse
import sys

def encode_latex_to_image(latex_str, display_mode=False):
    """Converts a raw LaTeX string into a CodeCogs SVG image Markdown link."""
    cleaned_latex = latex_str.strip()
    
    # Optional styling for CodeCogs SVG (e.g., set font size or color if needed)
    encoded_url = f"https://latex.codecogs.com/svg.image?{urllib.parse.quote(cleaned_latex)}"
    
    if display_mode:
        return f"\n\n<p align=\"center\">\n  <img src=\"{encoded_url}\" alt=\"{cleaned_latex}\" />\n</p>\n\n"
    else:
        return f"![{cleaned_latex}]({encoded_url})"

def convert_jekyll_to_medium(text):
    # 1. Remove YAML Front Matter (--- ... ---)
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)

    # 2. Convert Jekyll {% highlight python %} ... {% endhighlight %} to standard code blocks
    text = re.sub(
        r'\{%\s*highlight\s+(\w+)\s*%\}(.*?)\{%\s*endhighlight\s*%\}', 
        r'```\1\2```', 
        text, 
        flags=re.DOTALL
    )

    # 3. Strip remaining Jekyll Liquid tags like {% include ... %} or {{ site.baseurl }}
    text = re.sub(r'\{%.*?%\}', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)

    # 4. Convert Display Math ($$...$$)
    def handle_display_math(match):
        latex = match.group(1)
        return encode_latex_to_image(latex, display_mode=True)

    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', handle_display_math, text, flags=re.DOTALL)

    # 5. Convert Inline Math ($...$)
    def handle_inline_math(match):
        latex = match.group(1)
        return encode_latex_to_image(latex, display_mode=False)

    # Matches single $...$ while ignoring escaped \$ and display $$...$$
    text = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', handle_inline_math, text)

    return text.strip()

def process_file(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_medium{ext}"

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    medium_content = convert_jekyll_to_medium(raw_content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(medium_content)

    print(f"Successfully converted: {input_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_medium.py <path_to_post.md>")
        sys.exit(1)

    file_path = sys.argv[1]
    process_file(file_path)
