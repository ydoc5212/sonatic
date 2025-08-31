#!/usr/bin/env python3

import argparse
import os
import json
import csv
import re
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
except ImportError:
    pass  # dotenv is optional


# Utility functions for primary key generation
def generate_file_hash(file_path: str, length: int = 32) -> str:
    """Generate hash of file content for unique identification"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:length]


class DocumentValidator:
    """Configuration-driven data validation for extracted document fields."""
    
    def __init__(self, required_fields: list[str], optional_fields: list[str] = None, field_types: Dict[str, str] = None):
        """
        Initialize validator with field requirements.
        
        Args:
            required_fields: Fields that must be present and non-empty
            optional_fields: Fields that may be present (for documentation)
            field_types: Data type validation for specific fields (e.g., {"Total": "currency", "Date": "date"})
        """
        self.required_fields = required_fields
        self.optional_fields = optional_fields or []
        self.field_types = field_types or {}
    
    def _validate_currency(self, value: str) -> tuple[bool, str]:
        """Validate currency field (remove symbols, ensure numeric)"""
        if not isinstance(value, str):
            return False, "Currency must be text format"
        
        # Remove common currency symbols and whitespace
        clean_value = value.replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip()
        
        try:
            float_val = float(clean_value)
            if float_val < 0:
                return False, "Currency amount cannot be negative"
            return True, "Valid currency"
        except ValueError:
            return False, f"Invalid currency format: {value}"
    
    def _validate_date(self, value: str) -> tuple[bool, str]:
        """Basic date validation (ensure it looks like a date)"""
        if not isinstance(value, str):
            return False, "Date must be text format"
        
        # Simple check - contains date-like patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
            r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
            r'[A-Za-z]+ \d{1,2}, \d{4}',      # January 15, 2024
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, value):
                return True, "Valid date format"
        
        return False, f"Invalid date format: {value}"
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate extracted data against field requirements.
        
        Args:
            data: Extracted document data
            
        Returns:
            (success, message): Validation result and message
        """
        # Step 1: Check required fields are present
        missing_fields = []
        for field in self.required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Step 2: Validate field types (if specified)
        for field, expected_type in self.field_types.items():
            if field in data and data[field] is not None:
                if expected_type == "currency":
                    is_valid, msg = self._validate_currency(data[field])
                elif expected_type == "date":
                    is_valid, msg = self._validate_date(data[field])
                else:
                    continue  # Unknown type, skip validation
                
                if not is_valid:
                    return False, f"Invalid {expected_type} for {field}: {msg}"
        
        return True, "Validation passed"


class DocumentProcessor(ABC):
    # Extraction is task-focused and intentionally lossy - we only extract
    # the specific fields needed for the task at hand, not all document content
    @abstractmethod
    def get_extracted_fields(self) -> list[str]:
        pass
    
    @abstractmethod
    def extract_fields(self, content: str) -> Dict[str, Any]:
        pass


class DocumentAction(ABC):
    def __init__(self, primary_key_field: Optional[str] = None):
        """
        Initialize action with optional primary key for entity validation.
        
        Args:
            primary_key_field: Field name for business entity validation. None if no field-based validation.
        """
        self.primary_key_field = primary_key_field
    
    def generate_file_key(self, file_path: str) -> str:
        """Generate file hash key for file-level deduplication (always used)."""
        return generate_file_hash(file_path)
    
    def generate_primary_key(self, data: Dict[str, Any]) -> Optional[str]:
        """Generate business entity key for data quality validation (when available)."""
        if self.primary_key_field and self.primary_key_field in data:
            return str(data[self.primary_key_field])
        return None
    
    @abstractmethod
    def generate_filename(self, doc_type: str, data: Dict[str, Any], file_path: str, key: str) -> str:
        """Generate output filename. Override for different naming strategies."""
        pass
    
    def check_file_duplicates(self, file_key: str, output_path: str) -> bool:
        """Check if this file was already processed. Override for custom logic."""
        return False  # Default: no file-level deduplication
    
    def check_primary_key_duplicates(self, entity_key: Optional[str], output_path: str) -> bool:
        """Check if this business entity already exists. Override for custom logic."""
        return False  # Default: no primary key-level deduplication
    
    def check_duplicates(self, data: Dict[str, Any], file_path: str, key: str, output_path: str) -> bool:
        """Check if this should be skipped as duplicate. Override for custom logic."""
        return False  # Default: no deduplication (backward compatibility)
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate data meets action requirements. Returns (success, error_msg)"""
        pass
    
    @abstractmethod 
    def execute(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Execute action. Returns (success, result_msg)"""
        pass


class APInvoiceCSVAction(DocumentAction):
    def __init__(self):
        super().__init__(primary_key_field="Invoice #")
        self.validator = DocumentValidator(
            required_fields=["Vendor", "Invoice #", "Date Due", "Total"],
            field_types={
                "Total": "currency",
                "Date Due": "date"
            }
        )
    
    def generate_filename(self, doc_type: str, data: Dict[str, Any], file_path: str, key: str) -> str:
        """Generate compiled CSV file (compile pattern)"""
        return f"{doc_type}_log.csv"
    
    def check_file_duplicates(self, file_key: str, output_path: str) -> bool:
        """Check if file hash already exists in CSV (primary deduplication)"""
        return self._is_file_duplicate(file_key, output_path)
    
    def check_primary_key_duplicates(self, entity_key: Optional[str], output_path: str) -> bool:
        """Check if business key already exists in CSV (data quality validation)"""
        if entity_key is None:
            return False
        return self._is_entity_duplicate(entity_key, output_path)
    
    def check_duplicates(self, data: Dict[str, Any], file_path: str, key: str, output_path: str) -> bool:
        """Legacy method for backward compatibility"""
        file_key = self.generate_file_key(file_path)
        return self.check_file_duplicates(file_key, output_path)
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate AP Invoice data has all required fields"""
        return self.validator.validate(data)
    
    def _is_file_duplicate(self, file_key: str, csv_path: str) -> bool:
        """Check if file hash already exists in CSV file"""
        if not os.path.exists(csv_path):
            return False  # No file means no duplicates
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)  # Skip header row
                if not header:
                    return False  # Empty file
                
                # Find the File Hash column index
                try:
                    hash_index = header.index("File Hash")
                except ValueError:
                    return False  # File Hash column doesn't exist
                
                # Check each row for the file hash
                for row in reader:
                    if len(row) > hash_index and row[hash_index] == file_key:
                        return True  # Found duplicate
                        
            return False  # No duplicate found
            
        except Exception:
            return False  # Error reading file, assume no duplicate
    
    def _is_entity_duplicate(self, entity_key: str, csv_path: str) -> bool:
        """Check if business entity already exists in CSV file"""
        if not os.path.exists(csv_path):
            return False  # No file means no duplicates
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)  # Skip header row
                if not header:
                    return False  # Empty file
                
                # Find the column index for our primary key
                try:
                    key_index = header.index(self.primary_key_field)
                except ValueError:
                    return False  # Primary key column doesn't exist
                
                # Check each row for the entity key
                for row in reader:
                    if len(row) > key_index and row[key_index] == entity_key:
                        return True  # Found duplicate
                        
            return False  # No duplicate found
            
        except Exception:
            return False  # Error reading file, assume no duplicate
    
    def _is_duplicate(self, data: Dict[str, Any], file_path: str, csv_path: str) -> bool:
        """Legacy method: Check if primary key already exists in CSV file"""
        entity_key = self.generate_primary_key(data)
        if not entity_key:
            return False
        return self._is_entity_duplicate(entity_key, csv_path)
    
    def execute(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Append AP Invoice data to CSV using dual-key system"""
        file_key = self.generate_file_key(file_path)
        entity_key = self.generate_primary_key(data)
        csv_path = self.generate_filename("ap_invoice", data, file_path, file_key)
        
        # Primary check: File-level deduplication (robust)
        if self.check_file_duplicates(file_key, csv_path):
            return True, "duplicate"
        
        # Secondary check: Primary key-level validation (data quality)
        if entity_key and self.check_primary_key_duplicates(entity_key, csv_path):
            # Log entity duplication as a warning but don't block processing
            print(f"WARNING: Business entity {entity_key} already exists but from different file - {file_path}")
        
        try:
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(csv_path)
            
            # Define field order for CSV (include File Hash)
            field_order = ["Vendor", "Invoice #", "Date Due", "Total", "File Hash"]
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers if new file
                if not file_exists:
                    writer.writerow(field_order)
                
                # Write data row in correct order (include file hash)
                row = [data["Vendor"], data["Invoice #"], data["Date Due"], data["Total"], file_key]
                writer.writerow(row)
            
            return True, "appended"
            
        except Exception as e:
            return False, f"Failed to write to CSV: {str(e)}"


class PurchaseOrderJSONAction(DocumentAction):
    def __init__(self):
        super().__init__(primary_key_field="PO Number")
        self.validator = DocumentValidator(
            required_fields=["PO Number", "Vendor", "Order Date", "Delivery Date", "Total Amount"]
        )
    
    def generate_filename(self, doc_type: str, data: Dict[str, Any], file_path: str, key: str) -> str:
        """Generate individual JSON file using file key (transform pattern)"""
        return f"{doc_type}_{key}.json"
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate Purchase Order data has all required fields"""
        return self.validator.validate(data)
    
    def execute(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Export Purchase Order data to individual JSON file using dual-key system"""
        file_key = self.generate_file_key(file_path)
        entity_key = self.generate_primary_key(data)
        filename = self.generate_filename("purchase_order", data, file_path, file_key)
        
        # Check for duplicates (returns False by default for JSON files)
        if self.check_duplicates(data, file_path, file_key, filename):
            return True, "duplicate"
        
        # Generate structured JSON
        json_output = {
            "file_hash": file_key,
            "po_number": entity_key,  # Keep business key for data integrity
            "vendor": data["Vendor"],
            "dates": {
                "order_date": data["Order Date"],
                "delivery_date": data["Delivery Date"]
            },
            "financial": {
                "total_amount": data["Total Amount"]
            },
            "source_file": os.path.basename(file_path),
            "processed_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(json_output, f, indent=2)
            return True, "json_created"
        except Exception as e:
            return False, f"Failed to write JSON: {str(e)}"


class ReceiptJSONAction(DocumentAction):
    def __init__(self):
        super().__init__(primary_key_field=None)  # No business entity validation for receipts
        self.validator = DocumentValidator(
            required_fields=["Merchant", "Date", "Amount", "Payment Method"],
            optional_fields=["Category"]
        )
    
    def generate_filename(self, doc_type: str, data: Dict[str, Any], file_path: str, key: str) -> str:
        """Generate individual JSON file using hash (transform pattern)"""
        return f"{doc_type}_{key}.json"
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate Receipt data has all required fields"""
        return self.validator.validate(data)
    
    def execute(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Export Receipt data to individual JSON file using dual-key system"""
        file_key = self.generate_file_key(file_path)
        filename = self.generate_filename("receipt", data, file_path, file_key)
        
        # Check for duplicates (returns False by default for JSON files)
        if self.check_duplicates(data, file_path, file_key, filename):
            return True, "duplicate"
        
        # Generate structured JSON
        json_output = {
            "merchant": data["Merchant"],
            "date": data["Date"],
            "amount": data["Amount"],
            "payment_method": data["Payment Method"],
            "source_file": os.path.basename(file_path),
            "file_hash": file_key,
            "processed_timestamp": datetime.now().isoformat()
        }
        
        # Only include category if present in data (don't infer)
        if "Category" in data and data["Category"]:
            json_output["category"] = data["Category"]
        
        try:
            with open(filename, 'w') as f:
                json.dump(json_output, f, indent=2)
            return True, "json_created"
        except Exception as e:
            return False, f"Failed to write JSON: {str(e)}"


class BankStatementCSVAction(DocumentAction):
    def __init__(self):
        super().__init__(primary_key_field=None)  # No business entity validation for bank statements
        self.validator = DocumentValidator(
            required_fields=["Amount", "Date", "Description"]
        )
    
    def generate_filename(self, doc_type: str, data: Dict[str, Any], file_path: str, key: str) -> str:
        """Generate compiled CSV file (compile pattern)"""
        return f"{doc_type}_transactions.csv"
    
    def check_file_duplicates(self, file_key: str, output_path: str) -> bool:
        """Check if file hash already exists in CSV (primary deduplication)"""
        return self._is_file_duplicate(file_key, output_path)
    
    def check_duplicates(self, data: Dict[str, Any], file_path: str, key: str, output_path: str) -> bool:
        """Legacy method for backward compatibility"""
        file_key = self.generate_file_key(file_path)
        return self.check_file_duplicates(file_key, output_path)
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate Bank Statement data has all required fields"""
        return self.validator.validate(data)
    
    def _is_file_duplicate(self, file_key: str, csv_path: str) -> bool:
        """Check if file hash already exists in CSV file"""
        if not os.path.exists(csv_path):
            return False  # No file means no duplicates
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)  # Skip header row
                if not header:
                    return False  # Empty file
                
                # Find the File Hash column index
                try:
                    hash_index = header.index("File Hash")
                except ValueError:
                    return False  # File Hash column doesn't exist
                
                # Check each row for the file hash
                for row in reader:
                    if len(row) > hash_index and row[hash_index] == file_key:
                        return True  # Found duplicate
                        
            return False  # No duplicate found
            
        except Exception:
            return False  # Error reading file, assume no duplicate
    
    def _is_duplicate(self, data: Dict[str, Any], file_path: str, csv_path: str) -> bool:
        """Legacy method for backward compatibility"""
        file_key = self.generate_file_key(file_path)
        return self._is_file_duplicate(file_key, csv_path)
    
    def execute(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Append Bank Statement transaction to CSV using dual-key system"""
        file_key = self.generate_file_key(file_path)
        csv_path = self.generate_filename("bank_statement", data, file_path, file_key)
        
        # Check for file duplicates (primary deduplication)
        if self.check_file_duplicates(file_key, csv_path):
            return True, "duplicate"
        
        try:
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(csv_path)
            
            # Define field order for CSV (include File Hash)
            field_order = ["Date", "Amount", "Description", "File Hash"]
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers if new file
                if not file_exists:
                    writer.writerow(field_order)
                
                # Write data row in correct order (include file hash)
                row = [data["Date"], data["Amount"], data["Description"], file_key]
                writer.writerow(row)
            
            return True, "appended"
            
        except Exception as e:
            return False, f"Failed to write to CSV: {str(e)}"


def validate_json_response(response_text: str, expected_fields: list[str]) -> tuple[bool, Dict[str, Any], str]:
    """Validate JSON response and check for expected fields."""
    # Clean markdown formatting from response
    cleaned_text = response_text.strip()
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text[7:]  # Remove ```json
    if cleaned_text.startswith('```'):
        cleaned_text = cleaned_text[3:]   # Remove ```
    if cleaned_text.endswith('```'):
        cleaned_text = cleaned_text[:-3]  # Remove ```
    cleaned_text = cleaned_text.strip()
    
    try:
        data = json.loads(cleaned_text)
        if not isinstance(data, dict):
            return False, {}, "Response is not a JSON object"
        
        # Check all expected fields are present
        missing_fields = []
        for field in expected_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            return False, {}, f"Missing required fields: {', '.join(missing_fields)}"
        
        return True, data, "Success"
    except json.JSONDecodeError as e:
        return False, {}, f"Invalid JSON format: {str(e)}"


class GeminiClient:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def extract_fields(self, content: str, expected_fields: list[str]) -> Dict[str, Any]:
        errors = []
        
        for attempt in range(3):
            if attempt == 0:
                prompt = f"""Extract the following fields from this document:
{', '.join(expected_fields)}

Please return the response in valid JSON format.

Document content:
{content}

Return only a JSON object with the extracted values. Use null for missing fields. 
IMPORTANT: Return valid JSON format only."""
            else:
                prompt = f"""You must extract EXACTLY these fields from this document:
{', '.join(expected_fields)}

CRITICAL: Your response must be ONLY valid JSON. No explanations, no markdown formatting, no additional text.

Document content:
{content}

Return a JSON object with these exact field names: {', '.join(expected_fields)}
Use null for any missing values.
RESPOND WITH VALID JSON ONLY."""
            
            try:
                response = self.model.generate_content(prompt)
                is_valid, data, error_msg = validate_json_response(response.text, expected_fields)
                
                if is_valid:
                    return data
                else:
                    errors.append(f"Attempt {attempt + 1}: {error_msg}")
                    
            except Exception as e:
                errors.append(f"Attempt {attempt + 1}: API error - {str(e)}")
        
        # Print detailed error information
        print("ERROR: All 3 extraction attempts failed:")
        for error in errors:
            print(f"  - {error}")
        return None


class APInvoiceProcessor(DocumentProcessor):
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.action = APInvoiceCSVAction()
    
    def get_extracted_fields(self) -> list[str]:
        return ["Vendor", "Invoice #", "Date Due", "Total"]
    
    def extract_fields(self, content: str) -> Dict[str, Any]:
        return self.gemini_client.extract_fields(content, self.get_extracted_fields())
    
    def execute_action(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Execute the post-extraction action"""
        # First validate the data
        is_valid, msg = self.action.validate_data(data)
        if not is_valid:
            return False, f"Validation failed: {msg}"
        
        # Execute the action
        return self.action.execute(data, file_path)


class PurchaseOrderProcessor(DocumentProcessor):
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.action = PurchaseOrderJSONAction()
    
    def get_extracted_fields(self) -> list[str]:
        return ["PO Number", "Vendor", "Order Date", "Delivery Date", "Total Amount"]
    
    def extract_fields(self, content: str) -> Dict[str, Any]:
        return self.gemini_client.extract_fields(content, self.get_extracted_fields())
    
    def execute_action(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Execute the post-extraction action"""
        # First validate the data
        is_valid, msg = self.action.validate_data(data)
        if not is_valid:
            return False, f"Validation failed: {msg}"
        
        # Execute the action
        return self.action.execute(data, file_path)


class ReceiptProcessor(DocumentProcessor):
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.action = ReceiptJSONAction()
    
    def get_extracted_fields(self) -> list[str]:
        return ["Merchant", "Date", "Amount", "Payment Method", "Category"]
    
    def extract_fields(self, content: str) -> Dict[str, Any]:
        return self.gemini_client.extract_fields(content, self.get_extracted_fields())
    
    def execute_action(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Execute the post-extraction action"""
        # First validate the data
        is_valid, msg = self.action.validate_data(data)
        if not is_valid:
            return False, f"Validation failed: {msg}"
        
        # Execute the action
        return self.action.execute(data, file_path)


class BankStatementProcessor(DocumentProcessor):
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.action = BankStatementCSVAction()
    
    def get_extracted_fields(self) -> list[str]:
        return ["Amount", "Date", "Description"]
    
    def extract_fields(self, content: str) -> Dict[str, Any]:
        return self.gemini_client.extract_fields(content, self.get_extracted_fields())
    
    def execute_action(self, data: Dict[str, Any], file_path: str) -> tuple[bool, str]:
        """Execute the post-extraction action"""
        # First validate the data
        is_valid, msg = self.action.validate_data(data)
        if not is_valid:
            return False, f"Validation failed: {msg}"
        
        # Execute the action
        return self.action.execute(data, file_path)


class ProcessorRegistry:
    # TODO: Add more document types (contracts, etc.)
    _processors = {
        'ap_invoice': APInvoiceProcessor,
        'purchase_order': PurchaseOrderProcessor,
        'receipt': ReceiptProcessor,
        'bank_statement': BankStatementProcessor
    }
    
    @classmethod
    def get_processor(cls, doc_type: str, gemini_client: GeminiClient) -> Optional[DocumentProcessor]:
        processor_class = cls._processors.get(doc_type)
        if processor_class:
            try:
                return processor_class(gemini_client)
            except ValueError as e:
                print(f"Error initializing processor: {e}")
                return None
        return None
    
    @classmethod
    def get_available_types(cls) -> list[str]:
        return list(cls._processors.keys())


def read_file(file_path: str) -> str:
    # TODO: Support additional file types (.pdf, .docx, .jpg, .png, etc.)
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def main():
    parser = argparse.ArgumentParser(description='Ingest documents')
    parser.add_argument('files', nargs='+', help='Files to ingest')
    # TODO: Make --type optional with auto-detection of document type by default
    parser.add_argument('--type', dest='doc_type', required=True,
                        choices=ProcessorRegistry.get_available_types(),
                        help='Document type')
    
    args = parser.parse_args()
    
    # Initialize shared Gemini client once for all processors
    try:
        gemini_client = GeminiClient()
    except ValueError as e:
        print(f"ERROR: Failed to initialize Gemini client: {e}")
        return
    
    processed = 0
    duplicates = 0
    failed = 0
    
    for file_path in args.files:
        try:
            # Try to read the file first
            try:
                content = read_file(file_path)
            except FileNotFoundError:
                print(f"ERROR: File not found - {file_path}")
                failed += 1
                continue
            except ValueError as e:
                print(f"ERROR: Unsupported file type - {file_path} ({e})")
                failed += 1
                continue
            except Exception as e:
                print(f"ERROR: Could not read file - {file_path} ({e})")
                failed += 1
                continue
            
            # Try to get processor
            processor = ProcessorRegistry.get_processor(args.doc_type, gemini_client)
            if not processor:
                print(f"ERROR: Could not initialize processor for document type '{args.doc_type}' - {file_path}")
                failed += 1
                continue
                
            print(f"Processing {file_path} as {args.doc_type}")
            
            # Try extraction
            extracted_data = processor.extract_fields(content)
            
            if extracted_data is None:
                print(f"ERROR: Field extraction failed - {file_path}")
                failed += 1
            else:
                # Execute post-extraction action (single action per document type)
                # NOTE: Current design implements one action per processor for assignment requirements.
                # Architecture supports extending to action pipelines for production (e.g., csv + webform + email chain)
                # by modifying processors to hold action lists instead of single action instances.
                # TODO: Add --action CLI argument to support action selection (e.g., --action csv, --action json)
                if hasattr(processor, 'execute_action'):
                    action_success, action_msg = processor.execute_action(extracted_data, file_path)
                    if action_success:
                        if action_msg == "duplicate":
                            duplicates += 1
                        else:  # "appended" or other success
                            processed += 1
                    else:
                        print(f"ERROR: Action failed - {file_path}: {action_msg}")
                        failed += 1
                else:
                    # No action defined for this processor type
                    processed += 1
                
        except Exception as e:
            print(f"ERROR: Unexpected error processing {file_path}: {e}")
            failed += 1
    
    total_files = processed + duplicates + failed
    if total_files == 1:
        filename = args.files[0]  # Single file
        if failed > 0:
            print(f"\nSummary: Processing failed ({filename})")
        elif duplicates > 0:
            print(f"\nSummary: Duplicate skipped ({filename})")
        else:
            print(f"\nSummary: Successfully processed ({filename})")
    else:
        print(f"\nSummary: {total_files} files processed ({processed} new, {duplicates} duplicates, {failed} failed)")


if __name__ == '__main__':
    main()