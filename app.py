import streamlit as st
import random
from main import backend
import os
from io import BytesIO
import base64
from fpdf import FPDF
import io
import tempfile

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

object = backend()

def main():
    st.markdown("""
        <h1 style='text-align: left; font-size: 48px; margin-bottom: 20px;'>
            Intelligent Tax Document Processing and Extraction
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'summarised_text_chunks' not in st.session_state:
        st.session_state.summarised_text_chunks = []
    if 'extracted_project_code' not in st.session_state:
        st.session_state.extracted_project_code = ""
    if 'extracted_tax_year' not in st.session_state:
        st.session_state.extracted_tax_year = ""
    if 'extracted_jurisdiction' not in st.session_state:
        st.session_state.extracted_jurisdiction = ""
    if 'summary_to_display' not in st.session_state:
        st.session_state.summary_to_display = []
    if 'edited_summaries' not in st.session_state:
        st.session_state.edited_summaries = []

    with st.container():
        st.subheader("Upload a PDF File")
        uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
        submit_button = st.button("Submit")
    
    if submit_button:
        if uploaded_file is None:
            st.error("Please upload a PDF file before submitting.")
        else:
            with st.spinner("Processing..."):
                pdf_path = "uploaded.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())  
                
                if not os.path.exists(pdf_path):
                    st.error("File could not be saved. Please try again.")
                    return
                
                extracted_text = object.convert_to_image_and_extract(pdf_path)
                st.write("✅ Text extracted!")
                
                extracted_project_code, extracted_tax_year, extracted_jurisdiction = object.extract_fields(extracted_text)
                st.write("✅ Fields extracted!")
                
                summarised_text_chunks, scores = object.summarize(extracted_text)
                st.write("✅ Summary generated!")
                
                summary_to_display = object.display_poor_summary(summarised_text_chunks, scores)
                st.write("Oops, we need your help with some summaries->")
            
                st.session_state.processed = True
                st.session_state.summarised_text_chunks = summarised_text_chunks
                st.session_state.extracted_project_code = extracted_project_code
                st.session_state.extracted_tax_year = extracted_tax_year
                st.session_state.extracted_jurisdiction = extracted_jurisdiction
                st.session_state.summary_to_display = summary_to_display

                # ✅ Initialize edited summaries so they persist
                st.session_state.edited_summaries = summarised_text_chunks.copy()

    if st.session_state.processed:
        st.subheader("Edit Summarized Text")

        with st.form(key="edit_form"):
            for i in range(len(st.session_state.edited_summaries)):
                if i in st.session_state.summary_to_display:
                    # ✅ Persist previous edits instead of resetting on rerun
                    st.session_state.edited_summaries[i] = st.text_area(
                        f"Summary {i+1}",
                        value=st.session_state.edited_summaries[i],  # ✅ Keep previous edits
                        key=f"summary_{i}"
                    )

            submit_edits = st.form_submit_button("Generate PDF Report")

        if submit_edits:
            try:
                with st.spinner("Generating PDF..."):
                    generated_pdf_path = generate_pdf(
                        st.session_state.edited_summaries,
                        st.session_state.extracted_project_code, 
                        st.session_state.extracted_tax_year, 
                        st.session_state.extracted_jurisdiction
                    )

                    if generated_pdf_path and os.path.exists(generated_pdf_path):
                        with open(generated_pdf_path, "rb") as file:
                            generated_pdf_data = file.read()

                        st.download_button(
                            label="Download PDF Report",
                            data=generated_pdf_data,
                            file_name=f"{st.session_state.extracted_project_code}_{st.session_state.extracted_jurisdiction}_{st.session_state.extracted_tax_year}_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF generated successfully!")
                        os.remove(generated_pdf_path)
                    else:
                        st.error("Failed to generate PDF.")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
    
if __name__ == "__main__":
    main()
