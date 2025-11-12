#!/usr/bin/env python3
"""
PDF Text Extraction Script for KPZ Research Papers
Extracts text from PDF files and saves as readable text files
"""

import os
import pdfplumber
from pathlib import Path

def extract_pdf_text(pdf_path, output_path=None):
    """
    Extract text from a PDF file and save to a text file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Optional output path for text file
    
    Returns:
        str: Extracted text content
    """
    if output_path is None:
        # Create output filename by replacing .pdf with .txt
        output_path = pdf_path.replace('.pdf', '_extracted.txt')
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            
            print(f"Extracting text from: {os.path.basename(pdf_path)}")
            print(f"Number of pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"Processing page {page_num}/{len(pdf.pages)}...")
                
                # Extract text from the page
                page_text = page.extract_text()
                
                if page_text:
                    text_content.append(f"\n=== PAGE {page_num} ===\n")
                    text_content.append(page_text)
                    text_content.append("\n")
            
            # Join all text content
            full_text = "".join(text_content)
            
            # Save to text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"✅ Text extracted and saved to: {output_path}")
            print(f"Total characters extracted: {len(full_text)}")
            
            return full_text
            
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {str(e)}")
        return None

def main():
    """Extract text from all PDF files in the KPZ directory."""
    
    # Define the KPZ directory
    kpz_dir = Path(r"c:\Users\adamf\Desktop\University 2025\Experimental Physics\KPZ")
    
    # Find all PDF files
    pdf_files = list(kpz_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    print("\n" + "="*50)
    print("Starting text extraction...")
    print("="*50 + "\n")
    
    # Extract text from each PDF
    for pdf_file in pdf_files:
        extract_pdf_text(str(pdf_file))
        print("-" * 30)
    
    print("\n🎉 All PDF files processed!")
    print("\nText files created:")
    txt_files = list(kpz_dir.glob("*_extracted.txt"))
    for txt_file in txt_files:
        print(f"  - {txt_file.name}")

if __name__ == "__main__":
    main()