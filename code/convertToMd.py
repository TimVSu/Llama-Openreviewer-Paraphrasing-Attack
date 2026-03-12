import argparse
import json
import re

def json_to_markdown(json_path: str, output_path: str) -> None:
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    title = metadata.get("title", "Untitled Paper")
    authors = metadata.get("authors", [])
    sections = metadata.get("sections", [])

    def format_section(section: dict) -> str:
        heading = section.get("heading")
        text = re.sub(r'\n\d{3}\b', '', section.get("text", ""))
        return "\n".join(filter(None, [f"## {heading}" if heading else None, f"{text}\n"]))

    md_content = "\n".join(filter(None, [
        f"# {title}\n",
        f"**Authors:** {', '.join(authors)}\n" if authors else None,
        *map(format_section, sections)
    ]))

    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"Successfully converted to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSON paper file to Markdown.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output Markdown file")
    args = parser.parse_args()
    json_to_markdown(args.input, args.output)