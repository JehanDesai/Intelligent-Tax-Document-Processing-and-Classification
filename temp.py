import streamlit as st
from main import backend
import os
import time
from fpdf import FPDF
import tempfile

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'extracted_project_code' not in st.session_state:
    st.session_state.extracted_project_code = ""
if 'extracted_tax_year' not in st.session_state:
    st.session_state.extracted_tax_year = ""
if 'extracted_jurisdiction' not in st.session_state:
    st.session_state.extracted_jurisdiction = ""
if 'summarised_text_chunks' not in st.session_state:
    st.session_state.summarised_text_chunks = []
if 'summary_to_display' not in st.session_state:
    st.session_state.summary_to_display = []
if 'edited_summaries' not in st.session_state:
    st.session_state.edited_summaries = []


def generate_pdf(summarized_chunks, project_code, tax_year, jurisdiction):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 16)

    def clean_text(text):
        replacements = {
            '\u2018': "'", '\u2019': "'",  
            '\u201c': '"', '\u201d': '"',  
            '\u2013': '-', '\u2014': '--',  
            '\u2026': '...',               
        }
        for char, replacement in replacements.items():
            if isinstance(text, str):
                text = text.replace(char, replacement)
        return text
    
    pdf.cell(0, 10, f"Project Code: {project_code}", ln=True, align='C')
    pdf.cell(0, 10, f"Tax Year: {tax_year}", ln=True, align='C')
    pdf.cell(0, 10, f"Jurisdiction: {jurisdiction}", ln=True, align='C')
    
    pdf.line(10, pdf.get_y() + 5, 200, pdf.get_y() + 5)
    pdf.ln(10)  
    
    pdf.set_font('Arial', '', 12)
    for i, chunk in enumerate(summarized_chunks, 1):
        pdf.multi_cell(0, 8, f"{i}. {clean_text(chunk)}", align='L')
        pdf.ln(5)  
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file_path = temp_file.name
    temp_file.close()
    
    pdf.output(temp_file_path)
    
    return temp_file_path


def main():
    st.markdown("""
        <h1 style='text-align: left; font-size: 48px; margin-bottom: 20px;'>
            Intelligent Tax Document Processing and Extraction
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing
    
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    submit_button = st.button("Submit")
    
    if submit_button and uploaded_file:
        with st.spinner("Processing..."):
            pdf_path = "uploaded.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            object = backend()
            
            # Extract Text
            st.session_state.extracted_text = object.convert_to_image_and_extract(pdf_path)
            
            # Extract Fields
            st.session_state.extracted_project_code, st.session_state.extracted_tax_year, st.session_state.extracted_jurisdiction = object.extract_fields(st.session_state.extracted_text)
            
            # Summarize
            st.session_state.summarised_text_chunks, scores = object.summarize(st.session_state.extracted_text)
            st.session_state.summary_to_display = object.display_poor_summary(st.session_state.summarised_text_chunks, scores)
            
            if not st.session_state.edited_summaries:
                st.session_state.edited_summaries = st.session_state.summarised_text_chunks.copy()    
    # Grid Layout
    col1, col2 = st.columns([2, 1])
    if st.session_state.extracted_text:
        with col1:
            st.subheader("Extracted Text")
            st.text_area("Extracted Text", st.session_state.extracted_text, height=300, disabled=True, key="grid_extracted_text")
    if st.session_state.extracted_project_code:
        with col2:
            st.subheader("Document Details")
            st.markdown(f"**Project Code:** {st.session_state.extracted_project_code}")
            st.markdown(f"**Tax Year:** {st.session_state.extracted_tax_year}")
            st.markdown(f"**Jurisdiction:** {st.session_state.extracted_jurisdiction}")
    
    if st.session_state.summarised_text_chunks:
        st.markdown("---")
        st.subheader("Edit Summarized Text")
        for i in range(len(st.session_state.summarised_text_chunks)):
            if i in st.session_state.summary_to_display:
                new_text = st.text_area(f"Summary {i+1}", st.session_state.edited_summaries[i], key=f"summary_{i}")
                if new_text != st.session_state.edited_summaries[i]:
                    st.session_state.edited_summaries[i] = new_text
    
    if st.button("Generate PDF"):
        with st.spinner("Generating PDF..."):
            pdf_path = generate_pdf(
                st.session_state.edited_summaries,
                st.session_state.extracted_project_code,
                st.session_state.extracted_tax_year,
                st.session_state.extracted_jurisdiction
            )
            if os.path.exists(pdf_path):
                st.success("PDF Generated Successfully!")
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download Summarized PDF",
                        data=f.read(),
                        file_name="summarized_document.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("Failed to generate PDF. Please try again.")

if __name__ == "__main__":
    main()
