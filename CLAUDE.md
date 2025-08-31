# Document Ingestion Agent Architecture

## Overview
A modular document processing system that extracts structured data from unstructured documents and executes configurable follow-up actions. Built for extensibility across document types and action workflows.

## Core Architecture

### Document Processing Flow
```
Input Document → File Reader → Document Processor → Field Extraction → Action Execution → Output
```

### Key Components

#### 1. Document Processors (`DocumentProcessor` ABC)
- **Purpose**: Define document-specific field extraction logic
- **Interface**: 
  - `get_extracted_fields()` → Returns list of fields to extract via LLM
  - `extract_fields(content)` → Extracts structured data via LLM
- **Current Implementations**: 
  - `APInvoiceProcessor` (Vendor, Invoice #, Date Due, Total)
  - `ReceiptProcessor` (Merchant, Date, Amount, Payment Method, Category)
  - `PurchaseOrderProcessor` (PO Number, Vendor, Order Date, Delivery Date, Total Amount)
  - `BankStatementProcessor` (Amount, Date, Description)

#### 2. Document Actions (`DocumentAction` ABC) 
- **Purpose**: Define post-extraction workflows with sophisticated key and filename management
- **Core Interface**:
  - `generate_primary_key(data, file_path)` → Creates unique identifier (field-based or hash-based)
  - `generate_filename(doc_type, data, file_path, key)` → Creates output filename following naming philosophy
  - `check_duplicates(data, file_path, key, output_path)` → Detects existing records (overridable)
  - `validate_data(data)` → Validates extracted data completeness
  - `execute(data, file_path)` → Orchestrates the full action workflow
- **Current Implementations**: 
  - `APInvoiceCSVAction` (field-based key, compiled CSV)
  - `ReceiptJSONAction` (hash-based key, individual JSON files)
  - `PurchaseOrderJSONAction` (field-based key, individual JSON files)
  - `BankStatementCSVAction` (hash-based key, compiled CSV with hash column)

#### 3. Processor Registry
- **Purpose**: Maps document types to processor classes
- **Extensibility**: Add new document types by registering processors
- **Current**: `{'ap_invoice': APInvoiceProcessor, 'receipt': ReceiptProcessor, 'purchase_order': PurchaseOrderProcessor, 'bank_statement': BankStatementProcessor}`

#### 4. Gemini Integration (`GeminiClient`)
- **Purpose**: LLM-powered field extraction with retry logic
- **Features**: 
  - 3-attempt retry with escalating prompts
  - JSON response validation and markdown cleaning
  - Detailed error reporting per attempt

## Data Flow

1. **File Reading**: Support for .txt files (extensible to PDF, images)
2. **Document Detection**: Currently requires `--type`, planned auto-detection
3. **Field Extraction**: 
   - Processor defines expected fields
   - GeminiClient sends structured prompts to LLM
   - Response validated and cleaned
4. **Action Execution**:
   - Data validation against action requirements
   - Action-specific logic (CSV append, API calls, etc.)
   - Duplicate prevention (configurable per action)

## Extensibility Points

### Adding Document Types
```python
class ReceiptProcessor(DocumentProcessor):
    def get_expected_fields(self):
        return ["Merchant", "Date", "Amount", "Category"]
    
    def __init__(self):
        self.action = ReceiptJSONAction()
```

### Adding Action Types  
```python
class WebformAction(DocumentAction):
    def execute(self, data, file_path):
        # POST to web API, populate forms, etc.
```

### Adding File Types
Extend `read_file()` function to support PDF, images with OCR, etc.

## Current Features

### AP Invoice Processing
- **Fields**: Vendor, Invoice #, Date Due, Total
- **Action**: CSV export to `ap_invoice_log.csv`
- **Primary Key**: Field-based (Invoice #)
- **Duplicate Prevention**: Check existing CSV entries
- **File Pattern**: Compiled log (descriptive filename)

### Receipt Processing
- **Fields**: Merchant, Date, Amount, Payment Method, Category (optional)
- **Action**: Individual JSON export
- **Primary Key**: Hash-based (MD5 of file content)
- **Duplicate Prevention**: Overwrite existing files
- **File Pattern**: Transform single file (`receipt_[hash].json`)

### Purchase Order Processing
- **Fields**: PO Number, Vendor, Order Date, Delivery Date, Total Amount
- **Action**: Individual JSON export
- **Primary Key**: Field-based (PO Number)
- **Duplicate Prevention**: Overwrite existing files
- **File Pattern**: Transform single file (`purchase_order_[safe_key].json`)

### Bank Statement Processing
- **Fields**: Amount, Date, Description
- **Action**: CSV export to `bank_statement_transactions.csv` with File Hash column
- **Primary Key**: Hash-based (MD5 of file content)
- **Duplicate Prevention**: Check File Hash column in CSV
- **File Pattern**: Compiled log (descriptive filename)

### User Experience
- **Batch Processing**: Handle multiple files in single command
- **Quiet Success**: Only show errors, clean summaries
- **Smart Messaging**: Context-aware single-file vs batch feedback
- **Duplicate Handling**: Silent skip with summary reporting

### Architectural Patterns

#### Transform vs Compile Philosophy
- **Transform Pattern**: Single file → Single output (hash-based filenames)
  - Used by: Receipts, Purchase Orders
  - Filename: `{doc_type}_{key}.json`
  - Rationale: Direct transformation preserves file identity
- **Compile Pattern**: Multiple files → Compiled output (descriptive filenames)
  - Used by: AP Invoices, Bank Statements
  - Filename: `{doc_type}_log.csv` or `{doc_type}_transactions.csv`
  - Rationale: Aggregated data needs descriptive naming

#### Primary Key Strategies
- **Field-based**: Extract key from document data (e.g., Invoice #, PO Number)
- **Hash-based**: Use MD5 hash of file content for unique identification
- **Extensible**: `generate_primary_key()` method overridable for custom strategies

#### Deduplication Approaches
- **CSV Column Check**: Search existing CSV for matching keys/hashes
- **File Overwrite**: Replace existing JSON files (no deduplication)
- **Configurable**: `check_duplicates()` method customizable per action

## Technical Decisions

### Single Action Per Document Type
- **Rationale**: Assignment requires "a follow-up action" (singular)
- **Architecture**: Supports extending to action pipelines for production
- **Comment**: Code includes notes about scaling to action chains

### LLM Integration
- **Model**: Gemini 2.5 Pro for reliable structured extraction
- **Retry Logic**: Escalating prompt specificity over 3 attempts
- **Response Handling**: Markdown cleaning, JSON validation, field verification

### Error Strategy
- **Philosophy**: Unix-style (silent success, loud failures)
- **Categorization**: File errors, extraction errors, action errors
- **Recovery**: Continue processing remaining files on single failures

## Future Enhancements

### Planned Features
1. **Auto Document Detection**: LLM-based document type classification (eliminate `--type` requirement)
2. **PDF Support**: Text extraction and OCR capabilities for image-based documents
3. **Action Pipelines**: Chain multiple actions per document type (CLI argument to select action)
4. **Web Integration**: Form population, API submissions, webhook notifications
5. **Advanced File Types**: Email parsing, spreadsheet ingestion, image OCR
6. **Configuration Files**: External JSON/YAML configs for field definitions and action parameters

### Architecture Improvements
- **Plugin System**: Dynamic loading of processors/actions from external modules
- **Enhanced Decomposition**: Further separate `generate_filename` by document type parameter
- **Confidence Scoring**: LLM extraction confidence metrics and thresholds
- **Human-in-the-loop**: Review workflows for low-confidence extractions
- **Action Composition**: Support for action chains and conditional workflows
- **Multi-format Outputs**: Support CSV actions to optionally use hash-based filenames based on transform/compile context

### Discussed Extensibility Goals
- **Multiple Actions per Document**: Architecture supports extending to action selection via CLI
- **Hash-based CSV Filenames**: Future CSV actions may use transform pattern when appropriate
- **Composite Primary Keys**: Framework ready for complex key strategies (date+amount+description)
- **Enhanced Filename Generation**: `doc_type` parameter enables further filename customization
- **Utility Function Reuse**: `generate_file_hash` utility available for any hash-based needs

## Usage

```bash
# Single document (auto-detect type)
python3 ingestdoc.py invoice.txt

# Batch processing  
python3 ingestdoc.py *.txt

# Explicit document type
python3 ingestdoc.py receipt.txt --type receipt
python3 ingestdoc.py bank_statement.txt --type bank_statement
python3 ingestdoc.py purchase_order.txt --type purchase_order

# Environment setup
pip install google-generativeai python-dotenv
echo "GEMINI_API_KEY=your_key" > .env
```

## Project Structure
```
ingestdoc.py          # Main CLI and all classes
.env                  # API keys (gitignored)

# Test Documents
test*.txt             # Sample documents (invoices, receipts, bank statements, purchase orders)

# Output Files
ap_invoice_log.csv               # Compiled AP invoice data
bank_statement_transactions.csv  # Compiled bank statement data with File Hash column
receipt_[hash].json             # Individual receipt files (transform pattern)
purchase_order_[key].json       # Individual purchase order files (transform pattern)

# Documentation
CLAUDE.md            # This architecture document
```

## Class Hierarchy

```
DocumentProcessor (ABC)
├── APInvoiceProcessor
├── ReceiptProcessor  
├── PurchaseOrderProcessor
└── BankStatementProcessor

DocumentAction (ABC)
├── APInvoiceCSVAction (field-based key, compiled CSV)
├── ReceiptJSONAction (hash-based key, individual JSON)
├── PurchaseOrderJSONAction (field-based key, individual JSON)
└── BankStatementCSVAction (hash-based key, compiled CSV with hash column)

Utilities
├── generate_file_hash(file_path, length=32) → str
├── GeminiClient (LLM integration)
└── ProcessorRegistry (factory pattern)
```