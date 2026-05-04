import json
from fpdf import FPDF

def save_json_as_pdf(json_filename, pdf_filename):
    # 1. Load and prettify the JSON
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # indent=4 makes it "look as it is" with proper spacing
        pretty_json = json.dumps(data, indent=4)

    # 2. Setup PDF
    pdf = FPDF()
    pdf.add_page()
    # Using 'Courier' is critical for the "code look"
    pdf.set_font("Courier", size=10)
    
    # 3. Write the text
    # multi_cell handles line breaks and wraps text
    pdf.multi_cell(0, 5, pretty_json)
    
    # 4. Output
    pdf.output(pdf_filename)
    print(f"Success! {pdf_filename} created.")

save_json_as_pdf('dataset_card.json', 'dataset_card.pdf')