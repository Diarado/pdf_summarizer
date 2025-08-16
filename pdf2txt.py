#!/usr/bin/env python3
"""
PDF OCR Processing Script using EasyOCR
"""

from pathlib import Path
import fitz
import easyocr
import cv2
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFOCRProcessor:
    def __init__(self, pdf_folder="pdf_data", txt_folder="txt_data", languages=['en']):
        self.pdf_folder = Path(pdf_folder)
        self.txt_folder = Path(txt_folder)
        self.pdf_folder.mkdir(exist_ok=True)
        self.txt_folder.mkdir(exist_ok=True)
        
        logger.info(f"Initializing EasyOCR...")
        self.ocr = easyocr.Reader(languages, gpu=True)
        logger.info("Ready to process PDFs")
    
    def preprocess_image(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        except:
            return image
    
    def pdf_to_images(self, pdf_path, dpi_scale=2.0):
        images = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi_scale, dpi_scale)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                images.append(img)
        doc.close()
        return images
    
    def ocr_image(self, image, preprocess=True, confidence_threshold=0.3):
        if preprocess:
            image = self.preprocess_image(image)
        
        result = self.ocr.readtext(image, detail=1)
        if not result:
            return ""
        
        text_lines = []
        for detection in result:
            if len(detection) >= 3 and detection[2] >= confidence_threshold:
                text_lines.append(detection[1])
        
        return '\n'.join(text_lines)
    
    def process_single_pdf(self, pdf_path, preprocess_images=True, dpi_scale=2.0):
        images = self.pdf_to_images(pdf_path, dpi_scale)
        if not images:
            return False
        
        all_text = []
        empty_pages = 0
        
        for i, image in enumerate(images):
            logger.info(f"  Page {i+1}/{len(images)}")
            
            text = self.ocr_image(image, preprocess=preprocess_images)
            
            if not text.strip() and not preprocess_images:
                text = self.ocr_image(image, preprocess=True)
            
            if not text.strip():
                text = self.ocr_image(image, preprocess=True, confidence_threshold=0.1)
            
            if text.strip():
                all_text.append(f"\n{'='*50}\n=== Page {i+1} ===\n{'='*50}\n\n{text}\n")
            else:
                empty_pages += 1
        
        if all_text:
            output_path = self.txt_folder / f"{pdf_path.stem}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for: {pdf_path.name}\n")
                f.write(f"Total pages: {len(images)}\n")
                f.write(f"Pages with text: {len(images) - empty_pages}\n")
                f.write(f"Empty pages: {empty_pages}\n")
                f.write("="*60 + "\n\n")
                f.write('\n'.join(all_text))
            
            logger.info(f"✓ Saved: {output_path.name} ({len(images) - empty_pages}/{len(images)} pages)")
            return True
        
        return False
    
    def process_all_pdfs(self, preprocess_images=True, dpi_scale=2.0):
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        successful = 0
        for idx, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"\n[{idx}/{len(pdf_files)}] {pdf_path.name}")
            try:
                if self.process_single_pdf(pdf_path, preprocess_images, dpi_scale):
                    successful += 1
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info(f"\nComplete! {successful}/{len(pdf_files)} processed")
        logger.info(f"Output: {self.txt_folder.absolute()}")

def main():
    processor = PDFOCRProcessor(
        pdf_folder="pdf_data",
        txt_folder="txt_data",
        languages=['en']
    )
    
    processor.process_all_pdfs(
        preprocess_images=True,
        dpi_scale=2.0
    )

if __name__ == "__main__":
    main()