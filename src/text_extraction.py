import fitz
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, file_path: str) -> None:
        self._validate_file_path(file_path)
        self.file_path: str = file_path

    def _validate_file_path(self, path: str) -> None:
        if not path:
            raise ValueError("File path cannot be empty.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found at: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Path is not a valid file: {path}")

    def _initialize_result_dictionary(self) -> Dict[str, Any]:
        return {
            "status": "pending",
            "error_message": "",
            "metadata": {
                "page_count": 0,
                "title": "",
                "author": ""
            },
            "pages": [],
            "full_text": ""
        }

    def extract(self) -> Dict[str, Any]:
        result: Dict[str, Any] = self._initialize_result_dictionary()
        
        try:
            doc: fitz.Document = fitz.open(self.file_path)
            
            if doc.needs_pass:
                result["status"] = "error"
                result["error_message"] = "The PDF is encrypted and requires a password."
                doc.close()
                return result
                
            page_count: int = len(doc)
            if page_count == 0:
                result["status"] = "error"
                result["error_message"] = "The PDF contains zero pages."
                doc.close()
                return result

            doc_meta: Optional[Dict[str, Any]] = doc.metadata
            if doc_meta is not None:
                result["metadata"]["title"] = str(doc_meta.get("title", ""))
                result["metadata"]["author"] = str(doc_meta.get("author", ""))
            
            result["metadata"]["page_count"] = page_count
            
            extracted_pages: List[Dict[str, Any]] = []
            combined_text: List[str] = []
            
            for page_index in range(page_count):
                page: fitz.Page = doc.load_page(page_index)
                raw_text: str = page.get_text()
                clean_text: str = raw_text.strip()
                
                extracted_pages.append({
                    "page_number": page_index + 1,
                    "text": clean_text
                })
                
                if clean_text:
                    combined_text.append(clean_text)
                    
            if not combined_text:
                result["status"] = "error"
                result["error_message"] = "The PDF contains no extractable text."
                doc.close()
                return result
                
            result["pages"] = extracted_pages
            result["full_text"] = "\n\n".join(combined_text)
            result["status"] = "success"
            
            doc.close()
            
        except fitz.FileDataError as fde:
            logger.error(f"FileDataError for {self.file_path}: {fde}")
            result["status"] = "error"
            result["error_message"] = "The PDF file is corrupted or unreadable."
        except Exception as e:
            logger.error(f"Unexpected extraction error for {self.file_path}: {e}")
            result["status"] = "error"
            result["error_message"] = f"An unexpected error occurred: {str(e)}"
            
        return result
