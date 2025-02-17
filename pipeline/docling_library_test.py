#!/usr/bin/env python3
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
import logging
import os
import time
import ftfy
from pathlib import Path
from typing import Iterable

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # NVIDIA L4 (Ada Lovelace architecture)

USE_V2 = True

# Configure logging
log_file = Path('/mnt/data/unstructured/docling_test.log')
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for most verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)

_log = logging.getLogger(__name__)

def fix_greek_text(text: str) -> str:
    """Fix Greek text encoding issues using ftfy.
    
    This function uses ftfy to automatically detect and fix various text encoding issues,
    including Unicode escape sequences and other encoding problems.
    """
    return ftfy.fix_text(text)

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            # Export Docling document format to markdown
            markdown_content = conv_res.document.export_to_markdown()
            
            # Fix any Unicode issues in the markdown content
            fixed_content = fix_greek_text(markdown_content)
            
            # Write the fixed content to file
            with (output_dir / f"{doc_filename}.md").open("w", encoding='utf-8') as fp:
                fp.write(fixed_content)

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main(input_doc_paths=None):
    """Run docling conversion on the given input paths.
    
    Args:
        input_doc_paths: List of Path objects pointing to PDF files. If None, uses default test paths.
    """
    if input_doc_paths is None:
        input_doc_paths = list(Path("test_pdfs").glob("*.pdf"))

    accelerator_options = AcceleratorOptions(
        num_threads=10, device=AcceleratorDevice.CUDA
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    start_time = time.time()

    conv_results = converter.convert_all(
        input_doc_paths,
        raises_on_error=False,  # to let conversion run through all and examine results at the end
    )
    output_dir = Path("/mnt/data/unstructured/greek_processed_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=output_dir
    )

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The example failed converting {failure_count} on {len(input_doc_paths)}."
        )
        
if __name__ == "__main__":
    main()