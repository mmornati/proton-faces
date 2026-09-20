"""Binary route handlers: thumb, full, cover, crop.

These endpoints return image bytes. They use ``signed_or_token`` so they
work with either a bearer token or a short-lived signed URL. Public
access is gated by ``allow_public_thumbs()`` (F-01).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

import api  # noqa: E402  (intentional: see api_common docstring)
from auth import signed_or_token
from store import get_person, get_photo

log = logging.getLogger("api")

router = APIRouter()


@router.get("/api/photos/{uid}/thumb")
def api_thumb(uid: str, request: Request,
               _: object = Depends(signed_or_token)):
    """Serve a thumbnail (cached on disk)."""
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    return api._serve_thumb(uid, row)


@router.get("/api/photos/{uid}/full")
async def api_full(uid: str, request: Request,
                    _: object = Depends(signed_or_token)):
    """Serve the full-resolution image (cached on disk)."""
    row = get_photo(uid)
    if row is None:
        raise HTTPException(404, "photo not found")
    return await api._serve_full(uid, row, request)


@router.get("/api/people/{person_id}/cover")
def api_person_cover(person_id: int,
                      _: object = Depends(signed_or_token)):
    """Serve the cover image for a person cluster."""
    person = get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    return api._serve_person_cover(person_id, person)


@router.get("/api/faces/{face_id}/crop")
def api_face_crop(face_id: int,
                   _: object = Depends(signed_or_token)):
    """Serve the cropped face image."""
    face = api._face_row(face_id)
    if face is None:
        raise HTTPException(404, "face not found")
    return api._serve_face_crop(face_id, face)
