"""
NAKSHA Land Data Intelligence API

Current phase:
    - Database connectivity
    - Multi-source geospatial layer APIs
    - Layer metadata APIs
    - Parcel detail APIs
    - Basic spatial analytics
    - AOI / conflict / report analytics router

3D visualization is intentionally NOT handled here yet.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import geopandas as gpd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT CONFIGURATION
# ============================================================

_HERE = Path(__file__).resolve().parent

_ENV_CANDIDATES = [
    _HERE / ".env",
    _HERE.parent / ".env",
    _HERE.parent / "naksha-jaipur" / ".env",
    _HERE.parent.parent / "naksha-jaipur" / ".env",
]

for _candidate in _ENV_CANDIDATES:
    if _candidate.exists():
        load_dotenv(_candidate)
        break
else:
    load_dotenv()


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

DB_URL = os.getenv("DATABASE_URL")

TARGET_CRS = os.getenv(
    "TARGET_CRS",
    "EPSG:32643"
)

LOW_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "LOW_CONFIDENCE_THRESHOLD",
        "0.70"
    )
)

MATCH_SEARCH_DISTANCE_M = float(
    os.getenv(
        "MATCH_SEARCH_DISTANCE_M",
        "20"
    )
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        ""
    ).split(",")
    if origin.strip()
]


if not DB_URL:
    raise RuntimeError(
        "DATABASE_URL not found. Check your .env file."
    )


# ============================================================
# 3. DATABASE ENGINE
# ============================================================

engine = create_engine(DB_URL)


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NAKSHA Land Data Intelligence API",
    version="1.0.0"
)


# ============================================================
# 5. STORE SHARED CONFIGURATION
# ============================================================
#
# These values can be accessed by other routers/modules through:
#
#     request.app.state.low_confidence_threshold
#     request.app.state.match_search_distance_m
#
# This prevents different parts of the application from
# accidentally using different values.
# ============================================================

app.state.target_crs = TARGET_CRS
app.state.low_confidence_threshold = LOW_CONFIDENCE_THRESHOLD
app.state.match_search_distance_m = MATCH_SEARCH_DISTANCE_M


# ============================================================
# 6. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 7. ALLOWED TABLES
# ============================================================

SPATIAL_TABLES = {
    "parcels",
    "buildings",
    "roads",
    "utilities",
    "gnss_points",
    "ground_truth",
    "municipal_records",
    "ai_features",
    "building_changes",
}

TABULAR_TABLES = {
    "revenue_records",
    "data_sources",
    "conflicts",
}

ALL_UPLOADABLE_TABLES = (
    SPATIAL_TABLES |
    TABULAR_TABLES
)


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def _clean_for_json(df):
    """
    Convert datetime/date values into JSON-safe strings.

    Handles:
    - pandas datetime columns
    - Python datetime objects
    - Python date objects
    - pandas Timestamp objects
    """

    for col in df.columns:

        # Pandas datetime64 columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

        # Object columns may contain Python date/datetime objects
        elif pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].map(
                lambda value:
                    value.isoformat()
                    if isinstance(value, (datetime, date))
                    else value
            )

    return df


# ============================================================
# 9. BASIC SYSTEM ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "NAKSHA API is running",
        "target_crs": TARGET_CRS,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "match_search_distance_m": MATCH_SEARCH_DISTANCE_M,
    }


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "database": "connected"
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database unreachable: {e}"
        )


# ============================================================
# 10. OPTIONAL CONFIG ENDPOINT
# ============================================================

@app.get("/config")
def get_config():
    """
    Returns non-secret application configuration.

    Database credentials are NEVER returned.
    """

    return {
        "target_crs": TARGET_CRS,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "match_search_distance_m": MATCH_SEARCH_DISTANCE_M,
        "allowed_origins": ALLOWED_ORIGINS,
    }


# ============================================================
# 11. FILE UPLOAD
# ============================================================

@app.post("/upload/{table_name}")
async def upload_spatial_data(
    table_name: str,
    file: UploadFile = File(...)
):
    if table_name not in ALL_UPLOADABLE_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown table '{table_name}'"
        )

    suffix = os.path.splitext(
        file.filename
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(
            await file.read()
        )

        tmp_path = tmp.name

    try:
        gdf = gpd.read_file(tmp_path)

        if gdf.crs is None:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file has no CRS defined."
            )

        original_crs = str(gdf.crs)

        if gdf.crs.to_string() != TARGET_CRS:
            gdf = gdf.to_crs(TARGET_CRS)

        gdf = gdf.rename_geometry("geom")

        gdf.to_postgis(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False
        )

        return {
            "status": "success",
            "records_inserted": len(gdf),
            "target_table": table_name,
            "original_crs": original_crs,
            "transformed_crs": TARGET_CRS,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# 12. LAYER METADATA
# ============================================================

@app.get("/layers")
def list_all_layers():

    try:
        with engine.connect() as conn:

            result = conn.execute(
                text(
                    """
                    SELECT
                        layer_id,
                        name,
                        department,
                        purpose_tags,
                        default_visibility
                    FROM layers
                    """
                )
            )

            return {
                "layers": [
                    dict(row._mapping)
                    for row in result
                ]
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database Error: {e}"
        )


@app.get("/layers/recommended/{purpose}")
def get_recommended_layers(purpose: str):

    try:
        query = text(
            """
            SELECT
                layer_id,
                name,
                department,
                default_visibility
            FROM layers
            WHERE :purpose = ANY(purpose_tags)
            """
        )

        with engine.connect() as conn:

            result = conn.execute(
                query,
                {"purpose": purpose}
            )

            recommended = [
                dict(row._mapping)
                for row in result
            ]

        return {
            "purpose": purpose,
            "recommended_count": len(recommended),
            "layers": recommended,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database Error: {e}"
        )


# ============================================================
# 13. GEOJSON LAYER DATA
# ============================================================

@app.get("/layers/data/{table_name}")
def get_layer_geojson(
    table_name: str
):

    if table_name not in SPATIAL_TABLES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown or non-spatial table "
                f"'{table_name}'"
            )
        )

    try:

        sql = f"""
            SELECT
                *,
                ST_MakeValid(geom) AS corrected_geom
            FROM {table_name}
        """

        gdf = gpd.read_postgis(
            sql,
            con=engine,
            geom_col="corrected_geom",
            crs=TARGET_CRS
        )

        if "geom" in gdf.columns:
            gdf = gdf.drop(
                columns=["geom"]
            )

        gdf = _clean_for_json(gdf)

        # Convert to WGS84 for web maps
        gdf_web = gdf.to_crs("EPSG:4326")

        return json.loads(
            gdf_web.to_json()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Table '{table_name}' "
                f"error: {e}"
            )
        )


# ============================================================
# 14. TABULAR RECORDS
# ============================================================

@app.get("/records/{table_name}")
def get_table_records(
    table_name: str
):

    if table_name not in TABULAR_TABLES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown or spatial table "
                f"'{table_name}'"
            )
        )

    try:

        df = pd.read_sql(
            text(
                f"SELECT * FROM {table_name}"
            ),
            con=engine
        )

        df = _clean_for_json(df)

        return {
            "table": table_name,
            "count": len(df),
            "records": json.loads(
                df.to_json(
                    orient="records"
                )
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Table '{table_name}' "
                f"error: {e}"
            )
        )


# ============================================================
# 15. IMAGERY METADATA
# ============================================================

@app.get("/imagery")
def get_imagery_metadata():

    try:

        gdf = gpd.read_postgis(
            """
            SELECT *
            FROM imagery_metadata
            """,
            con=engine,
            geom_col="bounds",
            crs=TARGET_CRS
        )

        gdf = _clean_for_json(gdf)

        gdf_web = gdf.to_crs(
            "EPSG:4326"
        )

        return json.loads(
            gdf_web.to_json()
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"imagery_metadata "
                f"error: {e}"
            )
        )


# ============================================================
# 16. PARCEL DETAIL
# ============================================================

@app.get("/parcels/{parcel_id}")
def get_parcel_detail(
    parcel_id: str
):

    try:

        with engine.connect() as conn:

            parcel = conn.execute(
                text(
                    """
                    SELECT
                        parcel_id,
                        survey_number,
                        land_use,
                        owner_name,
                        area_sqm,
                        confidence_score
                    FROM parcels
                    WHERE parcel_id = :pid
                    """
                ),
                {"pid": parcel_id}
            ).mappings().first()

            if not parcel:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Parcel '{parcel_id}' "
                        f"not found"
                    )
                )

            buildings = conn.execute(
                text(
                    """
                    SELECT
                        building_id,
                        building_type,
                        height_m
                    FROM buildings
                    WHERE parcel_id = :pid
                    """
                ),
                {"pid": parcel_id}
            ).mappings().all()

            revenue = conn.execute(
                text(
                    """
                    SELECT *
                    FROM revenue_records
                    WHERE parcel_id = :pid
                    """
                ),
                {"pid": parcel_id}
            ).mappings().all()

            municipal = conn.execute(
                text(
                    """
                    SELECT
                        municipal_id,
                        upic,
                        ward_no,
                        property_type,
                        status
                    FROM municipal_records
                    WHERE parcel_id = :pid
                    """
                ),
                {"pid": parcel_id}
            ).mappings().all()

        return {
            "parcel": dict(parcel),
            "buildings": [
                dict(b)
                for b in buildings
            ],
            "revenue_records": [
                dict(r)
                for r in revenue
            ],
            "municipal_records": [
                dict(m)
                for m in municipal
            ],
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Parcel lookup error: {e}"
            )
        )


# ============================================================
# 17. ENCROACHMENT / INTERSECTION ANALYSIS
# ============================================================

@app.get("/analytics/encroachments")
def detect_encroachments():

    try:

        sql = """
            SELECT DISTINCT p.*
            FROM parcels p
            JOIN roads r
                ON ST_Intersects(
                    p.geom,
                    r.geom
                )
        """

        gdf = gpd.read_postgis(
            sql,
            con=engine,
            geom_col="geom",
            crs=TARGET_CRS
        )

        if gdf.empty:
            return {
                "status": "success",
                "encroachment_count": 0,
                "data": None
            }

        gdf_web = gdf.to_crs(
            "EPSG:4326"
        )

        return {
            "status": "success",
            "encroachment_count": len(gdf),
            "data": json.loads(
                gdf_web.to_json()
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis Error: {e}"
            )
        )


# ============================================================
# 18. BUFFER VIOLATIONS
# ============================================================

@app.get("/analytics/buffer-violation/{buffer_meters}")
def check_buffer_violations(
    buffer_meters: float
):

    try:

        query = text(
            """
            SELECT DISTINCT p.*
            FROM parcels p
            JOIN roads r
                ON ST_DWithin(
                    p.geom,
                    r.geom,
                    :buf
                )
            """
        )

        gdf = gpd.read_postgis(
            query,
            con=engine,
            geom_col="geom",
            params={
                "buf": buffer_meters
            },
            crs=TARGET_CRS
        )

        if gdf.empty:
            return {
                "status": "success",
                "violation_count": 0,
                "buffer_meters": buffer_meters,
                "data": None,
            }

        gdf_web = gdf.to_crs(
            "EPSG:4326"
        )

        return {
            "status": "success",
            "violation_count": len(gdf),
            "buffer_meters": buffer_meters,
            "data": json.loads(
                gdf_web.to_json()
            ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Buffer Analysis Error: {e}"
            )
        )


# ============================================================
# 19. AUDIT REPORT
# ============================================================

@app.get("/analytics/audit-report")
def generate_audit_report():

    try:

        with engine.connect() as conn:

            parcel_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM parcels"
                )
            ).scalar()

            building_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM buildings"
                )
            ).scalar()

            road_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM roads"
                )
            ).scalar()

            open_conflicts = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM conflicts
                    WHERE status = 'open'
                    """
                )
            ).scalar()

            pending_changes = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM building_changes
                    WHERE status = 'unreviewed'
                    """
                )
            ).scalar()

            encroachment_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT DISTINCT p.geom
                        FROM parcels p
                        JOIN roads r
                            ON ST_Intersects(
                                p.geom,
                                r.geom
                            )
                    ) AS sub
                    """
                )
            ).scalar()

        compliance_score = round(
            (
                (
                    parcel_count -
                    encroachment_count
                )
                /
                max(parcel_count, 1)
            ) * 100,
            2
        )

        return {
            "platform":
                "NAKSHA Land Data Intelligence - Jaipur",

            "audit_status":
                "Completed",

            "total_parcels":
                parcel_count,

            "total_buildings":
                building_count,

            "total_roads":
                road_count,

            "detected_encroachments":
                encroachment_count,

            "open_conflicts":
                open_conflicts,

            "pending_building_changes":
                pending_changes,

            "municipal_compliance_score_percent":
                compliance_score,

            "low_confidence_threshold":
                LOW_CONFIDENCE_THRESHOLD,

            "match_search_distance_m":
                MATCH_SEARCH_DISTANCE_M,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Audit Generation Error: {e}"
            )
        )


# ============================================================
# 20. PHASE 4 / 5 / 6 ANALYTICS ROUTER
# ============================================================
#
# IMPORTANT:
# main.py loads .env BEFORE this import.
#
# Therefore analysis_routes.py can safely use:
#
#     os.getenv("DATABASE_URL")
#
# and the same project configuration is available.
# ============================================================

from analysis_routes import router as analysis_router

app.include_router(
    analysis_router
)