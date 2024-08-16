import logging

import azure.functions as func
from bp_doc_intel_extract_city_names import bp_doc_intel_extract_city_names
from bp_doc_intel_only import bp_doc_intel_only
from bp_form_extraction_with_confidence import bp_form_extraction_with_confidence
from bp_pymupdf_extract_city_names import bp_pymupdf_extract_city_names
from bp_summarize_text import bp_summarize_text

# Reduce Azure SDK logging level
_logger = logging.getLogger("azure.core")
_logger.setLevel(logging.WARNING)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

app.register_blueprint(bp_form_extraction_with_confidence)
app.register_blueprint(bp_summarize_text)
app.register_blueprint(bp_doc_intel_extract_city_names)
app.register_blueprint(bp_pymupdf_extract_city_names)
app.register_blueprint(bp_doc_intel_only)
