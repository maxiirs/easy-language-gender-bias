import sys
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed")
    sys.exit(1)

def extract_summary(pdf_path, name):
    try:
        reader = PdfReader(pdf_path)
        print(f"\\n=== {name} ===")
        text_chunks = []
        for i in range(min(5, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_chunks.append(page_text)
                
        text = "\\n".join(text_chunks)
        print(text[:5000])
    except Exception as e:
        print(f"Error reading {name}: {e}")

pdf = "/Users/maximilian.buchmann/.gemini/antigravity/brain/31e764aa-f165-42e5-8f48-07243cbbdabb/.tempmediaStorage/53518197d5d487bf.pdf"
extract_summary(pdf, "arXiv 2510.04641")
