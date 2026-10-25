import os, sys
#### Force HuggingFace to copy files directly instead of creating Windows symlinks (To Prevent WinError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import logging
from pathlib import Path
from typing import Dict, Any
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class LayoutAwareIngestor:
    """
    Production Document Ingestor that preserves structural tables and layouts.
    - Uses IBM Docling for PDFs (layout & table structure recognition).
    - Uses Microsoft MarkItDown for Office spreadsheets/documents (.xlsx, .docx, .csv).
    """

    MAX_FILE_SIZE_MB: int = 20
    SUPPORTED_PDF_EXTS = {".pdf"}
    SUPPORTED_OFFICE_EXTS = {".xlsx", ".xls", ".docx", ".csv", ".pptx", ".txt", ".md"}

    def __init__(self, output_dir: str = "processed_md") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize engines lazily so startup is fast
        self._docling_converter: DocumentConverter | None = None
        self._markitdown_converter: MarkItDown | None = None

    @property
    def docling(self) -> DocumentConverter:
        """Lazy-loads Docling only when a PDF is ingested."""
        if self._docling_converter is None:
            logger.info("Initializing IBM Docling Layout Engine (first-run weights may download)...")
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    @property
    def markitdown(self) -> MarkItDown:
        """Lazy-loads MarkItDown only when an Office/Excel file is ingested."""
        if self._markitdown_converter is None:
            logger.info("Initializing Microsoft MarkItDown Engine...")
            self._markitdown_converter = MarkItDown()
        return self._markitdown_converter

    def _validate_file(self, file_path: Path) -> None:
        """Runs strict existence, extension, and 20MB size guardrails."""
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found at path: {file_path}")

        file_size_bytes = file_path.stat().st_size
        if file_size_bytes == 0:
            raise ValueError(f"File is completely empty (0 bytes): {file_path.name}")

        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File '{file_path.name}' is {file_size_mb:.2f} MB, which exceeds "
                f"the maximum allowed limit of {self.MAX_FILE_SIZE_MB} MB."
            )

        ext = file_path.suffix.lower()
        allowed = self.SUPPORTED_PDF_EXTS.union(self.SUPPORTED_OFFICE_EXTS)
        if ext not in allowed:
            raise ValueError(
                f"Unsupported file type '{ext}'. Allowed types: {sorted(list(allowed))}"
            )

    def ingest_file(self, file_path_str: str) -> Dict[str, Any]:
        """
        Parses a PDF or Office document into layout-preserved Markdown and saves it to disk.

        Returns a dictionary containing metadata and the raw Markdown string.
        """
        file_path = Path(file_path_str).resolve()
        self._validate_file(file_path)

        ext = file_path.suffix.lower()
        logger.info(f"Starting ingestion for '{file_path.name}' ({ext})...")

        try:
            if ext in self.SUPPORTED_PDF_EXTS:
                logger.info("Routing PDF through Docling for table-structure preservation...")
                conversion_result = self.docling.convert(str(file_path))
                markdown_text = conversion_result.document.export_to_markdown()
                engine_used = "IBM_Docling"
            else:
                logger.info("Routing Office/Spreadsheet file through MarkItDown...")
                conversion_result = self.markitdown.convert(str(file_path))
                markdown_text = conversion_result.text_content
                engine_used = "MS_MarkItDown"

            if not markdown_text or not markdown_text.strip():
                raise RuntimeError(
                    f"Extraction returned empty text for '{file_path.name}'. "
                    "Ensure the file is not corrupted or password-protected."
                )

            # Save extracted Markdown to processed_md/ for transparency and debugging
            output_md_path = self.output_dir / f"{file_path.stem}.md"
            output_md_path.write_text(markdown_text, encoding="utf-8")

            # Count how many Markdown table rows were preserved
            table_row_count = sum(
                1 for line in markdown_text.splitlines()
                if line.strip().startswith("|") and line.strip().endswith("|")
            )

            logger.info(
                f"Successfully ingested '{file_path.name}' via {engine_used}. "
                f"Preserved {table_row_count} table rows -> Saved to '{output_md_path}'"
            )

            return {
                "source_file": file_path.name,
                "engine": engine_used,
                "output_markdown_path": str(output_md_path),
                "char_count": len(markdown_text),
                "table_rows_detected": table_row_count,
                "markdown_content": markdown_text,
            }

        except Exception as e:
            logger.error(f"Failed to ingest '{file_path.name}': {str(e)}")
            raise RuntimeError(f"Ingestion failed for '{file_path.name}': {str(e)}") from e


# =====================================================================
# CLI TEST RUNNER (Run `python ingestion.py <path_to_file>` in terminal)
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python ingestion.py <path_to_pdf_or_excel_file>")
        print("Example: python ingestion.py data/sample_financials.pdf\n")
        sys.exit(1)

    target_file = sys.argv[1]
    ingestor = LayoutAwareIngestor(output_dir="processed_md")

    try:
        result = ingestor.ingest_file(target_file)
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"Source File          : {result['source_file']}")
        print(f"Parser Engine Used   : {result['engine']}")
        print(f"Total Characters     : {result['char_count']:,}")
        print(f"Table Rows Preserved : {result['table_rows_detected']}")
        print(f"Saved Markdown Path  : {result['output_markdown_path']}")
        print("=" * 60)
        print("\nPreview of first 500 characters:\n")
        print(result["markdown_content"][:500] + "\n...\n")
    except Exception as err:
        print(f"\n[ERROR] {err}\n")
        sys.exit(1)