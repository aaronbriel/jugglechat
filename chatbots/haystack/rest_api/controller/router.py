from fastapi import APIRouter

from rest_api.controller import file_upload
from rest_api.controller import search, feedback

"""
REFERENCE: Code initially pulled from https://github.com/deepset-ai/haystack
"""

router = APIRouter()

router.include_router(search.router, tags=["search"])
router.include_router(feedback.router, tags=["feedback"])
router.include_router(file_upload.router, tags=["file-upload"])
