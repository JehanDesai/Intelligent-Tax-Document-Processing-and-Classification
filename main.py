import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing import image
import re
from transformers import pipeline
from Levenshtein import distance as levenshtein_distance
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rouge_score import rouge_scorer
from fpdf import FPDF
import os
import tempfile
from model import Models


class backend:
    def preprocess_image(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.uint8(gray)

        # Binarization (Adaptive Thresholding)
        if len(gray.shape) == 2 and gray.dtype == np.uint8:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            raise ValueError("Grayscale conversion failed. Check image input.")

        # Deskew the images
        def deskew(image):
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]

            if angle == 90:
                return image
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 2:
                return image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
            return rotated

        deskewed = deskew(binary)

        # Removing noise using Gaussian blur and Median filtering
        denoised = cv2.medianBlur(deskewed, 3)
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

        # Enhance text edges using Morphological transformations
        kernel = np.ones((1,1), np.uint8)
        enhanced = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        return img, gray, binary, deskewed, denoised, enhanced



    def convert_to_image_and_extract(self, pdf_path):
        images = convert_from_path(pdf_path, dpi = 300)
        preprocessed_images = [self.preprocess_image(img) for img in images]

        # PSM 6 is good for uniform blocks of text
        custom_config = r'--oem 3 --psm 6'
        extracted_text = "\n".join([pytesseract.image_to_string(enhanced, config = custom_config) for _, _, _, _, _, enhanced in preprocessed_images])
        return extracted_text



    ner_pipeline = pipeline("ner", model = "dbmdz/bert-large-cased-finetuned-conll03-english")

    def extract_fields(self, text):
        project_code_pattern = r'\b(PC-\d{4,6}|Proj\d+)\b'
        tax_year_pattern = r'\b(20\d{2})\b'

        project_code = re.search(project_code_pattern, text)
        tax_year = re.search(tax_year_pattern, text)
        jurisdiction = self.extract_jurisdiction(text)

        return project_code.group(0) if project_code else "Project-KPMG", tax_year.group(0) if tax_year else "Year Not found", jurisdiction


    def extract_jurisdiction(self, text):
        ner_results = self.ner_pipeline(text)
        # print(ner_results)
        jurisdictions = [entity['word'] for entity in ner_results if entity['entity'] == 'B-LOC' or entity['entity'] == 'I-LOC']
        return " ".join(jurisdictions) if jurisdictions else "India"

    """
    Recursive v/s Other text splitters

    1. Mantains Context better
    2. Splits into hierarchy (sentences -> words -> characters)
    3. Reduces token loss
    4. Optimized for LLM's and Vector databases
    """

    def recursive_character_text_splitter(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # Max token size per chunk
        chunk_overlap=100  # Overlap to preserve context
    )
        text_chunks = text_splitter.split_text(text)
        return text_chunks



    def compute_levenshtein_distance(self, original, summarized):
        return 1 - (levenshtein_distance(original, summarized) / max(len(original), len(summarized)))



    def compute_rouge(self, original, summarized):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(original, summarized)
        return {k: v.fmeasure for k, v in scores.items()}


    """
        Encoder-Decoder Transformer (like T5)
        Trained on CNN/DailyMail dataset, optimized for abstractive summarization
        Great for long-text summarization
        Can be fine-tuned for text classification, Q&A, translation, etc.
    """
    def summarize(self, text):
        chunks = self.recursive_character_text_splitter(text)
        # GPT2_large_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-large")
        summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        scores = {}
        summarized_chunks = []
        for index, chunk in enumerate(chunks):
            chunk_summary = summarization_pipeline(chunk, max_length=75, min_length=50, do_sample=False)[0]['summary_text']
            summarized_chunks.append(chunk_summary)

            confidence_score = self.compute_levenshtein_distance(chunk, chunk_summary)
            rouge_score = self.compute_rouge(chunk, chunk_summary)
            
            scores[index] = {
                'confidence score' : confidence_score,
                'rouge score' : rouge_score
                }
            
        return summarized_chunks, scores


    def display_poor_summary(self, summary, scores):
        avg_rouge1_score = 0
        avg_rouge2_score = 0
        avg_rougeL_score = 0
        avg_confidence_score = 0

        for i in scores:
            avg_confidence_score += scores[i]['confidence score']
            avg_rouge1_score += scores[i]['rouge score']['rouge1']
            avg_rouge2_score += scores[i]['rouge score']['rouge2']
            avg_rougeL_score += scores[i]['rouge score']['rougeL']

        n = len(scores)
        avg_confidence_score = (avg_confidence_score + 0.7) / n
        avg_rouge1_score = (avg_rouge1_score + 0.7) / n
        avg_rouge2_score = (avg_rouge2_score + 0.7) / n
        avg_rougeL_score = (avg_rougeL_score + 0.7) / n
        # check = {}
        summary_to_display = []
        for i, s in enumerate(summary):
            if scores[i]['rouge score']['rouge1'] < avg_rouge1_score and scores[i]['rouge score']['rouge2'] < avg_rouge2_score and scores[i]['rouge score']['rougeL'] < avg_rougeL_score and scores[i]['confidence score'] < avg_confidence_score:
                summary_to_display.append(i)
        return summary_to_display
    
    def generate_pdf(self, summarized_chunks, project_code, tax_year, jurisdiction):
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