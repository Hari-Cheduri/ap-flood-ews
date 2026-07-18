"""
Canonical district registry used by the AP live flood-monitoring pipeline.

Coordinates are representative district/headquarters points used to request
one Sentinel-1 patch and one weather sequence per district. They are not
district boundary polygons.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class District(TypedDict):
    slug: str
    name: str
    latitude: float
    longitude: float


AP_DISTRICTS: List[District] = [
    {
        "slug": "alluri_sitharama_raju",
        "name": "Alluri Sitharama Raju",
        "latitude": 18.08,
        "longitude": 82.66,
    },
    {
        "slug": "anakapalli",
        "name": "Anakapalli",
        "latitude": 17.69,
        "longitude": 83.00,
    },
    {
        "slug": "ananthapuramu",
        "name": "Ananthapuramu",
        "latitude": 14.68,
        "longitude": 77.60,
    },
    {
        "slug": "annamayya",
        "name": "Annamayya",
        "latitude": 14.05,
        "longitude": 78.75,
    },
    {
        "slug": "bapatla",
        "name": "Bapatla",
        "latitude": 15.90,
        "longitude": 80.47,
    },
    {
        "slug": "chittoor",
        "name": "Chittoor",
        "latitude": 13.22,
        "longitude": 79.10,
    },
    {
        "slug": "east_godavari",
        "name": "East Godavari",
        "latitude": 16.99,
        "longitude": 81.78,
    },
    {
        "slug": "eluru",
        "name": "Eluru",
        "latitude": 16.71,
        "longitude": 81.10,
    },
    {
        "slug": "guntur",
        "name": "Guntur",
        "latitude": 16.31,
        "longitude": 80.44,
    },
    {
        "slug": "kakinada",
        "name": "Kakinada",
        "latitude": 16.99,
        "longitude": 82.25,
    },
    {
        "slug": "dr_br_ambedkar_konaseema",
        "name": "Dr. B.R. Ambedkar Konaseema",
        "latitude": 16.58,
        "longitude": 82.01,
    },
    {
        "slug": "krishna",
        "name": "Krishna",
        "latitude": 16.18,
        "longitude": 81.13,
    },
    {
        "slug": "kurnool",
        "name": "Kurnool",
        "latitude": 15.83,
        "longitude": 78.04,
    },
    {
        "slug": "nandyal",
        "name": "Nandyal",
        "latitude": 15.48,
        "longitude": 78.48,
    },
    {
        "slug": "ntr_vijayawada",
        "name": "NTR (Vijayawada)",
        "latitude": 16.51,
        "longitude": 80.65,
    },
    {
        "slug": "palnadu",
        "name": "Palnadu",
        "latitude": 16.24,
        "longitude": 80.05,
    },
    {
        "slug": "parvathipuram_manyam",
        "name": "Parvathipuram Manyam",
        "latitude": 18.78,
        "longitude": 83.43,
    },
    {
        "slug": "prakasam",
        "name": "Prakasam",
        "latitude": 15.50,
        "longitude": 80.05,
    },
    {
        "slug": "spsr_nellore",
        "name": "SPSR Nellore",
        "latitude": 14.44,
        "longitude": 79.99,
    },
    {
        "slug": "sri_sathya_sai",
        "name": "Sri Sathya Sai",
        "latitude": 14.17,
        "longitude": 77.81,
    },
    {
        "slug": "srikakulam",
        "name": "Srikakulam",
        "latitude": 18.29,
        "longitude": 83.90,
    },
    {
        "slug": "tirupati",
        "name": "Tirupati",
        "latitude": 13.63,
        "longitude": 79.42,
    },
    {
        "slug": "visakhapatnam",
        "name": "Visakhapatnam",
        "latitude": 17.69,
        "longitude": 83.22,
    },
    {
        "slug": "vizianagaram",
        "name": "Vizianagaram",
        "latitude": 18.10,
        "longitude": 83.40,
    },
    {
        "slug": "west_godavari",
        "name": "West Godavari",
        "latitude": 16.54,
        "longitude": 81.52,
    },
    {
        "slug": "ysr_kadapa",
        "name": "YSR Kadapa",
        "latitude": 14.47,
        "longitude": 78.82,
    },
]


DISTRICTS_BY_SLUG: Dict[str, District] = {
    district["slug"]: district for district in AP_DISTRICTS
}
