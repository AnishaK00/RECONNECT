"""Small authentication and patient-ownership API layer.

This module deliberately does not know about face or audio processing.
"""
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from database.db import (
    authenticate_actor,
    create_auth_session,
    create_caregiver,
    create_patient,
    get_caregiver_patients,
    get_session_actor,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class CaregiverRegistration(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    phone: str = Field(min_length=1)
    password: str = Field(min_length=6)
    caregiver_type: str = Field(min_length=1)
    family_relation: str | None = None
    profession: str | None = None


class PatientRegistration(BaseModel):
    caregiver_phone: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str = Field(min_length=6)
    dob: str = Field(min_length=1)
    phone: str = Field(min_length=1)
    email: str = Field(min_length=3)
    address: str = Field(min_length=1)
    medical_information: str = ""


class LoginRequest(BaseModel):
    actor_type: Literal["caregiver", "patient"]
    identifier: str = Field(min_length=1)
    password: str = Field(min_length=1)


def current_actor(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    actor = get_session_actor(authorization.removeprefix("Bearer ").strip())
    if not actor:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return actor


def require_caregiver(actor=Depends(current_actor)):
    if actor["actor_type"] != "caregiver":
        raise HTTPException(status_code=403, detail="Caregiver access is required.")
    return actor


@router.post("/caregivers/register", status_code=201)
def register_caregiver(payload: CaregiverRegistration):
    try:
        caregiver_id = create_caregiver(**payload.model_dump())
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="Email or phone is already registered.") from exc
        raise
    return {"id": caregiver_id}


@router.post("/patients/register", status_code=201)
def register_patient(payload: PatientRegistration):
    try:
        patient_id = create_patient(**payload.model_dump())
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="Username is already registered.") from exc
        raise
    if patient_id is None:
        raise HTTPException(status_code=422, detail="No caregiver account exists for that phone number.")
    return {"id": patient_id, "patient_code": f"PT{patient_id:04d}"}


@router.post("/login")
def login(payload: LoginRequest):
    actor = authenticate_actor(payload.actor_type, payload.identifier, payload.password)
    if not actor:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_auth_session(payload.actor_type, actor["id"])
    return {"token": token, "actor_type": payload.actor_type, "actor_id": actor["id"], "name": actor.get("name") or actor.get("full_name")}


@router.get("/me")
def me(actor=Depends(current_actor)):
    return actor


@router.get("/patients")
def caregiver_patients(actor=Depends(require_caregiver)):
    return {"patients": get_caregiver_patients(actor["actor_id"])}
