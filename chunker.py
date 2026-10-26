import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level = logging.INFO,
    format= "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S"
)
logger = logging.getLogger(__name__)

class DualModeChunker:
    """
    Production-grade document chunker designed for enterprise tables and narrative text.
    - Tables: Preserves schema by pinning column headers and parent headings to every sub-table.
    - Prose: Sliding-window token/word overlap anchored by hierarchical section headers.
    """

    def __init__(
        self,
        table_rows_per_chunk: int = 25,
        table_row_overlap: int = 2,
        prose_chunk_words: int = 350,
        prose_overlap_words: int =70,
        output_dir: str = "chunks"
    ) -> None:
        
        self.table_rows_per_chunk = table_rows_per_chunk
        self.table_row_overlap = table_row_overlap
        self.prose_chunk_words = prose_chunk_words
        self.prose_overlap_words = prose_overlap_words
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_table_line(self, line: str) -> bool:
        """Determines if a line is part of a Markdown table grid."""
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")
    
    def _is_table_separator(self, line: str) -> bool:
        """Determines if a line is the Markdown syntax separator (| --- | --- |)."""
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            return False
        # Matches patterns like | --- | :---: | --- |
        cleaned = stripped.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
        return len(cleaned) == 0

    def _extract_heading(self, line: str) -> Optional[str]:
        """Extracts Markdown headings (# Section Title)."""
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            return match.group(0)
        return None
    
    def chunk_markdown(self, markdown_text: str, source_filename: str) -> List[Dict[str, Any]]:
        """
        Parses Markdown text and returns a list of contextual, self-contained chunks.
        """
        lines = markdown_text.splitlines()
        chunks: List[Dict[str, Any]] = []
        
        current_heading = "Document Root"
        current_prose_accumulator: List[str] = []
        
        # State tracking for tables
        in_table = False
        table_headers: List[str] = []  # Line 0: Column names, Line 1: Separators
        table_data_rows: List[str] = []

        chunk_counter = 0

        def flush_prose():
            nonlocal chunk_counter, current_prose_accumulator
            if not current_prose_accumulator:
                return

            full_prose_text = "\n".join(current_prose_accumulator).strip()
            words = full_prose_text.split()
            current_prose_accumulator = []

            if not words:
                return

            # Apply sliding window chunking to prose
            step = self.prose_chunk_words - self.prose_overlap_words
            if step <= 0:
                step = self.prose_chunk_words

            for i in range(0, len(words), step):
                chunk_words = words[i:i + self.prose_chunk_words]
                if not chunk_words:
                    continue

                chunk_body = " ".join(chunk_words)
                # Programmatically prepend the hierarchical heading
                final_content = f"{current_heading}\n\n{chunk_body}".strip()

                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"{source_filename}_chunk_{chunk_counter:05d}",
                    "source": source_filename,
                    "type": "prose",
                    "section_heading": current_heading,
                    "content": final_content,
                    "word_count": len(chunk_words),
                    "estimated_tokens": int(len(chunk_words) * 1.3)
                })

        def flush_table():
            nonlocal chunk_counter, in_table, table_headers, table_data_rows
            if not table_headers or not table_data_rows:
                in_table = False
                table_headers = []
                table_data_rows = []
                return

            header_block = "\n".join(table_headers)
            total_rows = len(table_data_rows)
            step = self.table_rows_per_chunk - self.table_row_overlap
            if step <= 0:
                step = self.table_rows_per_chunk

            for start_idx in range(0, total_rows, step):
                end_idx = min(start_idx + self.table_rows_per_chunk, total_rows)
                row_slice = table_data_rows[start_idx:end_idx]

                if not row_slice:
                    break

                body_block = "\n".join(row_slice)
                
                # CRITICAL ARCHITECTURAL STEP:
                # Prepend both Section Heading AND Table Column Headers to every sub-chunk
                final_table_chunk = (
                    f"{current_heading}\n"
                    f"[Table Segment: Rows {start_idx + 1} to {end_idx} of {total_rows}]\n\n"
                    f"{header_block}\n"
                    f"{body_block}"
                )

                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"{source_filename}_chunk_{chunk_counter:05d}",
                    "source": source_filename,
                    "type": "table_segment",
                    "section_heading": current_heading,
                    "row_range": f"{start_idx + 1}-{end_idx}",
                    "total_table_rows": total_rows,
                    "content": final_table_chunk,
                    "word_count": len(final_table_chunk.split()),
                    "estimated_tokens": int(len(final_table_chunk.split()) * 1.3)
                })

                if end_idx == total_rows:
                    break

            in_table = False
            table_headers = []
            table_data_rows = []

        # Line-by-line streaming parse
        for line in lines:
            heading = self._extract_heading(line)
            if heading:
                if in_table:
                    flush_table()
                else:
                    flush_prose()
                current_heading = heading
                continue

            if self._is_table_line(line):
                # We encountered a table line
                if not in_table:
                    # Transitioning from prose to table
                    flush_prose()
                    in_table = True
                    table_headers.append(line)
                else:
                    # Already inside table
                    if len(table_headers) == 1 and self._is_table_separator(line):
                        table_headers.append(line)
                    else:
                        table_data_rows.append(line)
            else:
                # Regular line / non-table line
                if in_table:
                    if line.strip() == "":
                        # Empty line can signify end of table or gap
                        flush_table()
                    else:
                        flush_table()
                        current_prose_accumulator.append(line)
                else:
                    current_prose_accumulator.append(line)

        # Flush any remaining items in the buffer
        if in_table:
            flush_table()
        else:
            flush_prose()

        logger.info(
            f"Chunking finished for '{source_filename}': Generated {len(chunks)} contextual chunks."
        )
        return chunks

    def process_markdown_file(self, md_path_str: str) -> str:
        """Reads a .md file, applies dual-mode chunking, and saves chunks as JSON."""
        md_path = Path(md_path_str).resolve()
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        markdown_text = md_path.read_text(encoding="utf-8")
        chunks = self.chunk_markdown(markdown_text, source_filename=md_path.stem)

        out_json_path = self.output_dir / f"{md_path.stem}_chunks.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chunks)} chunks to '{out_json_path}'")
        return str(out_json_path)


# CLI TEST RUNNER
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python chunker.py <path_to_markdown_file>")
        print("Example: python chunker.py processed_md/sample_document.md\n")
        sys.exit(1)

    input_file = sys.argv[1]
    chunker = DualModeChunker(
        table_rows_per_chunk=25,
        table_row_overlap=2,
        prose_chunk_words=350,
        prose_overlap_words=70,
        output_dir="chunks"
    )

    try:
        json_output = chunker.process_markdown_file(input_file)
        
        # Load and print inspectable sample of chunk #1 and chunk #2
        with open(json_output, "r", encoding="utf-8") as f:
            generated_chunks = json.load(f)

        print("\n" + "=" * 65)
        print("CHUNKING COMPLETE - SUMMARY")
        print("=" * 65)
        print(f"Total Chunks Generated : {len(generated_chunks):,}")
        print(f"Output JSON Path       : {json_output}")
        print("=" * 65)

        if generated_chunks:
            print("\nINSPECTING CHUNK 1 (Table Header + First Rows):")
            print("-" * 65)
            print(generated_chunks[0]["content"][:600] + "\n...")

            if len(generated_chunks) > 1:
                print("\nINSPECTING CHUNK 2 (Notice the re-injected headers!):")
                print("-" * 65)
                print(generated_chunks[1]["content"][:600] + "\n...")

    except Exception as err:
        logger.error(f"Error during chunking: {err}")
        sys.exit(1)