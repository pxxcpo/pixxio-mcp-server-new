#!/usr/bin/env python3
"""
pixx.io MCP Server — Search & Filter
Supports: Claude Desktop (stdio), ChatGPT / Claude Web / Copilot (Streamable HTTP)

Environment variables:
  PIXXIO_API_KEY   — pixx.io API authentication token
  PIXXIO_BASE_URL  — Base URL, e.g. https://yourspace.px.media
  TRANSPORT        — "stdio" or "http" (default: "http")
  PORT             — HTTP port (default: 8000)
  HOST             — HTTP host (default: "0.0.0.0")
"""

import os
import json
import logging
from typing import Optional

import httpx
import base64
from fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pixxio-mcp")

# ── Configuration ─────────────────────────────────────────────────────────────
PIXXIO_API_KEY  = os.environ.get("PIXXIO_API_KEY", "")
PIXXIO_BASE_URL = os.environ.get("PIXXIO_BASE_URL", "").rstrip("/")
TRANSPORT       = os.environ.get("TRANSPORT", "http").lower()
PORT            = int(os.environ.get("PORT", "8000"))
HOST            = os.environ.get("HOST", "0.0.0.0")

# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "pixx.io DAM",
    instructions=(
        "pixx.io ist ein Digital Asset Management System (DAM). "
        "Nutze diese Tools um digitale Assets (Bilder, Videos, Dokumente) zu suchen und zu verwalten.\n\n"

        "SUCHANFRAGEN ZERLEGEN:\n"
        "Analysiere jede Suchanfrage und extrahiere folgende Komponenten:\n"
        "- Personen ('von Richard', 'mit Anna') → person_name Parameter\n"
        "- Zeitangaben ('letztes Jahr', '2024', 'diesen Monat') → date_from + date_to (als YYYY-MM-DD berechnen)\n"
        "- Events/Themen ('Sommerfest', 'Messeauftritt') → query Parameter\n"
        "- Dateityp ('Bilder'→image, 'Videos'→video, 'Audiodateien'→audio) → file_type Parameter\n"
        "- Ausrichtung ('Querformat'→landscape, 'Hochformat'→portrait) → orientation Parameter\n"
        "- Qualität ('hochwertig', 'beste', '5 Sterne') → rating_min=4 oder 5\n"
        "- Farbprofil ('für den Druck', 'CMYK') → colorspace='CMYK'\n"
        "- Ordner/Abteilungen → zuerst list_directories aufrufen → directory_id setzen\n"
        "- Sammlungen/Alben → zuerst list_collections aufrufen → collection_id setzen\n\n"

        "SEMANTIC SUCHE (semantic=True):\n"
        "Nutze dies bei visuell-beschreibenden Anfragen wie 'rotes Auto auf einer Straße' oder "
        "'Person am Strand bei Sonnenuntergang'. Ebenfalls als Fallback wenn die Standardsuche "
        "keine oder unpassende Ergebnisse liefert. "
        "Semantic kann mit allen anderen Filtern (file_type, orientation, rating_min etc.) kombiniert werden.\n\n"

        "CUSTOM METADATA SUCHE:\n"
        "Bei spezifischen Werten die kein allgemeiner Suchbegriff sind (z.B. 'Artikelnummer 12345', "
        "'Projektnummer ABC', 'Saison Winter 2024') → erst get_searchable_metadata_fields aufrufen "
        "→ passendes Feld finden → metadata_filters in search_assets nutzen.\n\n"

        "BEISPIELE:\n"
        "'Bilder von Richard vom Sommerfest letztes Jahr'\n"
        "→ query='Sommerfest', person_name='Richard', file_type='image', "
        "date_from='2025-01-01', date_to='2025-12-31'\n\n"
        "'Hochauflösende Querformat-Bilder für den Druck'\n"
        "→ file_type='image', orientation='landscape', colorspace='CMYK', "
        "sort_by='pixel', sort_direction='desc'\n\n"
        "'Rotes Auto auf einer Straße in höchster Auflösung'\n"
        "→ query='rotes Auto auf einer Straße', semantic=True, file_type='image', "
        "sort_by='pixel', sort_direction='desc'\n\n"
        "'Alle Bilder mit der Artikelnummer 12345'\n"
        "→ erst get_searchable_metadata_fields → Feld 'Artikelnummer' finden "
        "→ search_assets(metadata_filters=[{'field_id': <id>, 'value': '12345'}])"
    ),
)

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get_client() -> httpx.AsyncClient:
    """Create an authenticated async HTTP client for pixx.io."""
    if not PIXXIO_BASE_URL:
        raise ValueError("PIXXIO_BASE_URL ist nicht konfiguriert.")
    if not PIXXIO_API_KEY:
        raise ValueError("PIXXIO_API_KEY ist nicht konfiguriert.")
    return httpx.AsyncClient(
        base_url=PIXXIO_BASE_URL,
        headers={"Authorization": f"Bearer {PIXXIO_API_KEY}"},
        timeout=30.0,
    )


async def _api_get(path: str, params: Optional[dict] = None) -> dict:
    async with _get_client() as client:
        resp = await client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _resolve_person_ids(name: str) -> list[int]:
    """Resolve a person name to pixx.io person IDs using face recognition data.

    Searches files for the given name, extracts face/person objects, and
    returns IDs where firstName or lastName matches.
    """
    try:
        data = await _api_get("/api/v1/files", {
            "showFiles": "true",
            "pageSize": 50,
            "filter": json.dumps({
                "filterType": "searchTerm",
                "term": name,
                "useSynonyms": False,
            }),
            "responseFields": "id,faces",
        })
        person_ids: set[int] = set()
        name_lower = name.lower()
        for f in data.get("files", []):
            for face in (f.get("faces") or []):
                person = face.get("person") or {}
                first = (person.get("firstName") or "").lower()
                last  = (person.get("lastName") or "").lower()
                if name_lower in first or name_lower in last:
                    pid = person.get("id")
                    if pid:
                        person_ids.add(int(pid))
        logger.info(f"Person '{name}' → IDs: {person_ids or 'keine gefunden'}")
        return list(person_ids)
    except Exception as exc:
        logger.warning(f"Person-Auflösung für '{name}' fehlgeschlagen: {exc}")
        return []


def _absolute_url(url: str) -> str:
    """Ensure a URL is absolute by prepending PIXXIO_BASE_URL if it's a relative path."""
    if url and not url.startswith("http"):
        return f"{PIXXIO_BASE_URL}{url}"
    return url


def _build_filter(filters: list[dict], operator: str) -> Optional[dict]:
    """Combine a list of filter dicts into a single pixx.io filter object."""
    active = [f for f in filters if f]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    connector = "connectorAnd" if operator.upper() != "OR" else "connectorOr"
    return {"filterType": connector, "filters": active}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: search_assets
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def search_assets(
    query: str = "",
    semantic: bool = False,
    file_type: Optional[str] = None,
    file_extension: Optional[str] = None,
    format_type: Optional[str] = None,
    colorspace: Optional[str] = None,
    orientation: Optional[str] = None,
    rating_min: Optional[int] = None,
    person_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    min_pixel: Optional[int] = None,
    max_pixel: Optional[int] = None,
    directory_id: Optional[int] = None,
    collection_id: Optional[int] = None,
    metadata_filters: Optional[list] = None,
    operator: str = "AND",
    inverted_filters: Optional[list] = None,
    sort_by: str = "uploadDate",
    sort_direction: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """Search for assets in the pixx.io Digital Asset Management system.

    Supports two search modes:
    - Standard (default): keyword/text search across names, descriptions, keywords
    - Semantic (semantic=True): AI vector search for visual/descriptive queries

    All filters can be freely combined. The server translates parameters into
    the pixx.io filter API automatically.

    Args:
        query: Search term. Standard mode searches names, descriptions, keywords.
               With semantic=True: describe a visual scene, e.g. "red car on a street".
        semantic: Use AI semantic vector search. Best for visual/scene descriptions
                  or as fallback when standard search returns poor results.
        file_type: Filter by type: "image", "video", "audio".
        file_extension: Filter by extension: "jpg", "png", "pdf", "mp4", "tiff", "webp", etc.
        format_type: Filter by format type: "raw", "vector", "pdf", "web-image", "office",
                     "archive", "audio", "video", "image", "imageconverted".
        colorspace: Filter by colorspace: "RGB", "CMYK", or "GRAY". Use "CMYK" for print assets.
        orientation: Filter by orientation: "landscape", "portrait", or "square".
        rating_min: Minimum star rating (1–5). Use 4 or 5 for "high quality" requests.
        person_name: Person's name to filter by (uses face recognition). The server
                     automatically resolves the name to person IDs. Falls back to
                     keyword search if no face recognition match is found.
        date_from: Upload date start as YYYY-MM-DD (e.g. "2025-01-01").
                   Compute from natural language: "letztes Jahr" → "2025-01-01".
        date_to: Upload date end as YYYY-MM-DD (e.g. "2025-12-31").
        min_width: Minimum image width in pixels.
        min_height: Minimum image height in pixels.
        min_pixel: Minimum total pixel count (width × height).
        max_pixel: Maximum total pixel count.
        directory_id: Filter to this directory (includes subdirectories).
                      Use list_directories first to find the ID.
        collection_id: Filter to this collection/album.
                       Use list_collections first to find the ID.
        metadata_filters: Custom metadata field filters. List of objects:
                          {"field_id": <int>, "value": "<str>", "edit_type": "<text|date|selection>"}.
                          Use get_searchable_metadata_fields to discover field IDs and types.
        operator: How to combine filters: "AND" (default, all must match) or
                  "OR" (any must match).
        inverted_filters: Parameter names to negate, e.g. ["file_type"] excludes that type.
                          Valid names: query, file_type, file_extension, format_type,
                          colorspace, orientation, rating_min, person_name, date_from,
                          date_to, min_width, min_height, min_pixel, directory_id, collection_id.
        sort_by: "uploadDate" (default), "createDate", "modifyDate", "fileName",
                 "rating", "pixel", "width", "height", "id".
        sort_direction: "desc" (default, newest/largest first) or "asc".
        page: Page number, starts at 1.
        page_size: Results per page, max 100, default 20.

    Returns:
        Dictionary with 'ids', 'results' (asset summaries), 'total_results', page info,
        and 'search_mode' indicating which search mode was used.
    """
    inverted_set = set(inverted_filters or [])
    filters: list[dict] = []

    # ── Text / Semantic ────────────────────────────────────────────────────────
    if query and not semantic:
        filters.append({
            "filterType": "searchTerm",
            "term": query,
            "useSynonyms": True,
            "inverted": "query" in inverted_set,
        })

    # ── File type & format ─────────────────────────────────────────────────────
    if file_type:
        filters.append({
            "filterType": "fileType",
            "fileType": file_type,
            "inverted": "file_type" in inverted_set,
        })

    if file_extension:
        filters.append({
            "filterType": "fileExtension",
            "fileExtension": file_extension,
            "inverted": "file_extension" in inverted_set,
        })

    if format_type:
        filters.append({
            "filterType": "formatType",
            "formatType": format_type,
            "inverted": "format_type" in inverted_set,
        })

    if colorspace:
        filters.append({
            "filterType": "colorspace",
            "colorspace": colorspace,
            "inverted": "colorspace" in inverted_set,
        })

    # ── Visual properties ──────────────────────────────────────────────────────
    if orientation:
        filters.append({
            "filterType": "orientation",
            "orientation": orientation,
            "inverted": "orientation" in inverted_set,
        })

    if rating_min is not None:
        filters.append({
            "filterType": "rating",
            "rating": rating_min,
            "inverted": "rating_min" in inverted_set,
        })

    # ── Person (face recognition) ──────────────────────────────────────────────
    if person_name:
        person_ids = await _resolve_person_ids(person_name)
        if person_ids:
            if len(person_ids) == 1:
                filters.append({
                    "filterType": "person",
                    "personID": person_ids[0],
                    "inverted": "person_name" in inverted_set,
                })
            else:
                # Multiple persons match the name → OR-combine
                filters.append({
                    "filterType": "connectorOr",
                    "filters": [
                        {"filterType": "person", "personID": pid}
                        for pid in person_ids
                    ],
                    "inverted": "person_name" in inverted_set,
                })
        else:
            # Fallback: keyword filter
            logger.info(f"Kein Person-Match für '{person_name}', nutze keyword-Filter")
            filters.append({
                "filterType": "keyword",
                "term": person_name,
                "exactMatch": False,
                "inverted": "person_name" in inverted_set,
            })

    # ── Date range ─────────────────────────────────────────────────────────────
    if date_from or date_to:
        date_filter: dict = {
            "filterType": "uploadDate",
            "inverted": ("date_from" in inverted_set or "date_to" in inverted_set),
        }
        if date_from:
            date_filter["dateMin"] = f"{date_from} 00:00:00"
        if date_to:
            date_filter["dateMax"] = f"{date_to} 23:59:59"
        filters.append(date_filter)

    # ── Dimensions ─────────────────────────────────────────────────────────────
    if min_width is not None:
        filters.append({
            "filterType": "width",
            "min": min_width,
            "inverted": "min_width" in inverted_set,
        })

    if min_height is not None:
        filters.append({
            "filterType": "height",
            "min": min_height,
            "inverted": "min_height" in inverted_set,
        })

    if min_pixel is not None or max_pixel is not None:
        pf: dict = {
            "filterType": "pixel",
            "inverted": ("min_pixel" in inverted_set or "max_pixel" in inverted_set),
        }
        if min_pixel is not None:
            pf["min"] = min_pixel
        if max_pixel is not None:
            pf["max"] = max_pixel
        filters.append(pf)

    # ── Organisation ───────────────────────────────────────────────────────────
    if directory_id:
        filters.append({
            "filterType": "directory",
            "directoryID": directory_id,
            "includeSubdirectories": True,
            "inverted": "directory_id" in inverted_set,
        })

    if collection_id:
        filters.append({
            "filterType": "collection",
            "collectionID": collection_id,
            "inverted": "collection_id" in inverted_set,
        })

    # ── Custom metadata ────────────────────────────────────────────────────────
    for mf in (metadata_filters or []):
        field_id  = mf.get("field_id")
        value     = str(mf.get("value", ""))
        edit_type = mf.get("edit_type", "text")

        if edit_type == "date":
            filters.append({
                "filterType": "metadataFieldDate",
                "customMetadataFieldID": field_id,
                "dateMin": f"{value} 00:00:00",
                "dateMax": f"{value} 23:59:59",
            })
        elif edit_type in ("selection", "multiselection"):
            filters.append({
                "filterType": "metadataFieldSelection",
                "customMetadataFieldID": field_id,
                "term": value,
            })
        else:
            # text, number, and all other types
            filters.append({
                "filterType": "metadataFieldText",
                "customMetadataFieldID": field_id,
                "term": value,
                "exactMatch": False,
            })

    # ── Build API request ──────────────────────────────────────────────────────
    params: dict = {
        "showFiles": "true",
        "page": page,
        "pageSize": min(page_size, 100),
        "sortBy": sort_by,
        "sortDirection": sort_direction,
        "responseFields": (
            "id,fileName,fileExtension,fileType,previewFileURL,"
            "description,keywords,subject,rating,uploadDate,"
            "fileSize,width,height,orientation"
        ),
    }

    if semantic and query:
        params["semanticQuery"] = query

    combined_filter = _build_filter(filters, operator)
    if combined_filter:
        params["filter"] = json.dumps(combined_filter)

    logger.info(
        f"search_assets: mode={'semantic' if semantic and query else 'standard'}, "
        f"filter_count={len(filters)}, query={query!r}"
    )

    data     = await _api_get("/api/v1/files", params)
    files    = data.get("files", [])
    quantity = data.get("quantity", 0)

    ids: list[str] = []
    results: list[dict] = []
    for f in files:
        fid = str(f.get("id", ""))
        ids.append(fid)
        results.append({
            "id":             fid,
            "title":          f.get("fileName", ""),
            "description":    f.get("description") or f.get("subject", ""),
            "file_type":      f.get("fileType", ""),
            "file_extension": f.get("fileExtension", ""),
            "keywords":       f.get("keywords", []),
            "rating":         f.get("rating"),
            "preview_url":    _absolute_url(f.get("previewFileURL", "")),
            "upload_date":    f.get("uploadDate", ""),
            "file_size":      f.get("fileSize"),
            "dimensions":     f"{f['width']}×{f['height']}" if f.get("width") else None,
            "orientation":    f.get("orientation", ""),
        })

    return {
        "ids":           ids,
        "results":       results,
        "total_results": quantity,
        "page":          page,
        "page_size":     page_size,
        "search_mode":   "semantic" if (semantic and query) else "standard",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1b: get_preview
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def get_preview(id: str, width: int = 800):
    """Display an asset's preview image inline in the chat.

    Downloads the image server-side and returns it as base64 so it can be
    displayed directly in Claude Desktop without sandbox restrictions.

    Args:
        id: Asset ID (as returned by search_assets).
        width: Preview width in pixels (default: 800).

    Returns:
        The preview image displayed inline, plus a fallback URL.
    """
    data = await _api_get(f"/api/v1/files/{id}/convert", {
        "downloadType": "preview",
        "responseType": "path",
        "maxSize": width,
    })
    download_url = data.get("downloadURL") or data.get("downloadUrl") or ""

    if not download_url:
        raise ValueError(f"Kein Preview verfügbar für Asset {id}.")

    download_url = _absolute_url(download_url)
    fallback_text = f"Preview für Asset {id}: {download_url}"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(download_url)
            resp.raise_for_status()
            img_bytes = resp.content

        if not img_bytes:
            raise ValueError("Leere Bild-Antwort")

        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        mime_type = "image/png" if "png" in content_type else "image/jpeg"

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.info(f"get_preview: {len(img_bytes)} bytes, mime={mime_type}, asset={id}")
        return [
            ImageContent(type="image", data=img_b64, mimeType=mime_type),
            TextContent(type="text", text=fallback_text),
        ]

    except Exception as e:
        logger.warning(f"get_preview: Download fehlgeschlagen: {e}")
        return fallback_text


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: fetch_asset
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def fetch_asset(id: str) -> dict:
    """Fetch details and important metadata for a specific asset by its ID.

    Returns standard fields plus the importantMetadata fields configured
    in the DAM by the administrator.

    Args:
        id: Asset ID (as returned by search_assets).

    Returns:
        Full asset record including important metadata and face/person data.
    """
    data = await _api_get(f"/api/v1/files/{id}", {
        "responseFields": (
            "id,fileName,fileExtension,fileType,description,subject,"
            "keywords,rating,uploadDate,createDate,modifyDate,fileSize,"
            "width,height,orientation,colorspace,directory,faces,importantMetadata"
        ),
    })
    f = data.get("file", data)

    return {
        "id":                 str(f.get("id", id)),
        "file_name":          f.get("fileName", ""),
        "file_extension":     f.get("fileExtension", ""),
        "file_type":          f.get("fileType", ""),
        "file_size":          f.get("fileSize"),
        "description":        f.get("description", ""),
        "subject":            f.get("subject", ""),
        "keywords":           f.get("keywords", []),
        "rating":             f.get("rating"),
        "width":              f.get("width"),
        "height":             f.get("height"),
        "orientation":        f.get("orientation", ""),
        "colorspace":         f.get("colorspace", ""),
        "upload_date":        f.get("uploadDate", ""),
        "create_date":        f.get("createDate", ""),
        "modify_date":        f.get("modifyDate", ""),
        "directory":          f.get("directory"),
        "faces":              f.get("faces", []),
        "important_metadata": f.get("importantMetadata"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: list_directories
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def list_directories(
    parent_id: Optional[int] = None,
    show_tree: bool = False,
) -> dict:
    """List directories (folders) in the pixx.io DAM.

    Call this before searching to find the directory_id for a specific folder.

    Args:
        parent_id: ID of the parent directory. Omit for root level.
        show_tree: Return the full directory tree (all levels at once).

    Returns:
        List of directories with IDs, names, paths, and file counts.
    """
    if show_tree:
        data = await _api_get("/api/v1/directories/tree")
    else:
        params: dict = {}
        if parent_id:
            params["parentID"] = parent_id
        data = await _api_get("/api/v1/directories", params)

    dirs = data.get("directories", data.get("tree", []))

    def _fmt(d: dict) -> dict:
        result: dict = {
            "id":           d.get("id"),
            "name":         d.get("name", ""),
            "path":         d.get("path", ""),
            "has_children": d.get("hasChildren", False),
            "file_count":   d.get("quantity", 0),
        }
        if d.get("children"):
            result["children"] = [_fmt(c) for c in d["children"]]
        return result

    return {
        "directories": [_fmt(d) for d in dirs] if isinstance(dirs, list) else dirs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: list_collections
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def list_collections(
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """List collections (albums/lightboxes) in the pixx.io DAM.

    Call this before searching to find the collection_id for a specific album.

    Args:
        page: Page number.
        page_size: Results per page (max 100).

    Returns:
        List of collections with IDs, names, descriptions, and file counts.
    """
    data = await _api_get("/api/v1/collections", {
        "page": page,
        "pageSize": min(page_size, 100),
    })

    return {
        "collections": [
            {
                "id":          c.get("id"),
                "name":        c.get("name", ""),
                "description": c.get("description", ""),
                "is_dynamic":  c.get("isDynamic", False),
                "file_count":  c.get("filesQuantity", 0),
                "create_date": c.get("createDate", ""),
            }
            for c in data.get("collections", [])
        ],
        "total": data.get("quantity", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: get_keywords
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def get_keywords(
    query: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
) -> dict:
    """List keywords (tags) used across assets in the DAM.

    Use this to discover available keywords before formulating a search.

    Args:
        query: Optional text to filter keywords by name (wildcard search).
        page: Page number.
        page_size: Results per page (max 100).

    Returns:
        List of keywords with IDs and names.
    """
    params: dict = {
        "page": page,
        "pageSize": min(page_size, 100),
        "sortBy": "name",
        "sortDirection": "asc",
    }
    if query:
        params["name"] = query

    data = await _api_get("/api/v1/keywords", params)

    return {
        "keywords": [
            {"id": k.get("id"), "name": k.get("name", "")}
            for k in data.get("keywords", [])
        ],
        "total": data.get("quantity", 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6: get_searchable_metadata_fields
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(annotations={"readOnlyHint": True})
async def get_searchable_metadata_fields() -> dict:
    """List all custom metadata fields that can be searched (isSearchable=true).

    Use this when a query contains a specific value that might match a custom
    metadata field, e.g. "Artikelnummer 12345" or "Projektnummer ABC".

    After calling this tool, use the returned field_id and edit_type in the
    metadata_filters parameter of search_assets.

    Returns:
        List of searchable metadata fields with field_id, name, edit_type,
        and available selection_terms (for selection fields).
    """
    data = await _api_get("/api/v1/metadataFields", {
        "isSearchable": "true",
        "page": 0,
        "pageSize": 100,
        "sortBy": "name",
        "sortDirection": "asc",
    })

    return {
        "fields": [
            {
                "field_id":        f.get("id"),
                "name":            f.get("name", ""),
                "edit_type":       f.get("editType", "text"),
                "type":            f.get("type", "custom"),
                "selection_terms": f.get("selectionTerms", []),
            }
            for f in data.get("metadataFields", [])
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Health check endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for deployment monitoring."""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "server": "pixx.io MCP Server",
        "transport": TRANSPORT,
        "configured": bool(PIXXIO_BASE_URL and PIXXIO_API_KEY),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(f"Starting pixx.io MCP Server (transport={TRANSPORT})")
    logger.info(f"pixx.io: {PIXXIO_BASE_URL or 'NOT CONFIGURED'}")
    logger.info(f"API key: {'configured' if PIXXIO_API_KEY else 'NOT CONFIGURED'}")

    if TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    else:
        logger.info(f"HTTP server on {HOST}:{PORT}/mcp")
        mcp.run(transport="http", host=HOST, port=PORT, stateless_http=True)
