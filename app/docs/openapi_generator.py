# app/docs/openapi_generator.py
"""
OpenAPI/Swagger Documentation Generator
Auto-generates API documentation from FastAPI routes
"""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def generate_openapi_spec(app: FastAPI, output_file: str = "api_documentation.json"):
    """
    Generate OpenAPI specification from FastAPI app

    Args:
        app: FastAPI application instance
        output_file: Output file path for JSON spec
    """

    # Generate OpenAPI schema
    openapi_schema = get_openapi(
        title="SeeDeep.AI API Documentation",
        version="2.0.0",
        description="""
# SeeDeep.AI REST API & WebSocket Documentation

Real-time multi-camera object detection and tracking system with alerts.

## Features
- 📹 Camera Management
- 🎯 Real-time Object Detection
- 🔍 Object Tracking
- 🚨 Alert System (Speed, Tracking, Distance)
- 📧 Email Notifications
- 🎥 PTZ Camera Control
- 📡 WebSocket Streaming

## Authentication
Currently no authentication required. Add authentication headers when implemented.

## Rate Limiting
No rate limits currently enforced.

## WebSocket Endpoint
- **URL**: `ws://{host}:{port}/ws/camera/{camera_id}`
- **Protocol**: WebSocket
- **Purpose**: Real-time video stream with detections, tracking, and alerts

## Base URL
- **Development**: `http://localhost:8000`
- **Production**: Update with your production URL
        """,
        routes=app.routes,
    )

    # Enhance WebSocket documentation
    if "paths" not in openapi_schema:
        openapi_schema["paths"] = {}

    # Add WebSocket endpoint documentation
    openapi_schema["paths"]["/ws/camera/{camera_id}"] = {
        "get": {
            "tags": ["WebSocket"],
            "summary": "Camera WebSocket Stream",
            "description": """
# Real-time Camera Stream with Detections, Tracking, and Alerts

Connect to this WebSocket endpoint to receive real-time video frames with:
- Object detections from multiple models
- Object tracking with IDs
- Speed calculations
- Distance measurements
- **Alerts** (speed, tracking, distance)

## Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/camera/{camera_id}');

ws.onopen = () => {
    console.log('Connected to camera stream');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Handle detections, tracking, and alerts
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from camera stream');
};
```

## Message Format
See the schema below for the complete message structure.

## Frame Rate
Controlled by camera's configured FPS (default: 15 fps).

## Alerts
Alerts are included in the same WebSocket message when triggered:
- Speed alerts: When object speed exceeds/falls below threshold
- Tracking alerts: When object stays in frame longer/shorter than threshold
- Distance alerts: When object gets closer/farther than threshold
            """,
            "parameters": [
                {
                    "name": "camera_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Camera ID to stream from",
                }
            ],
            "responses": {
                "101": {
                    "description": "WebSocket connection established",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/WebSocketResponse"
                            },
                            "examples": {
                                "with_alerts": {
                                    "summary": "Message with alerts",
                                    "value": {
                                        "camera_id": "cam_001",
                                        "timestamp": 1704106800000,
                                        "results": {
                                            "general_detection": {
                                                "detections": [
                                                    {
                                                        "x1": 100.0,
                                                        "y1": 200.0,
                                                        "x2": 300.0,
                                                        "y2": 400.0,
                                                        "confidence": 0.95,
                                                        "class_id": 0,
                                                        "label": "person",
                                                    }
                                                ],
                                                "count": 1,
                                                "model": "general_detection",
                                            },
                                            "tracking": {
                                                "tracked_objects": {
                                                    "track_42": {
                                                        "track_id": "track_42",
                                                        "class_name": "person",
                                                        "bbox": [100, 200, 300, 400],
                                                        "centroid": [200, 300],
                                                        "confidence": 0.95,
                                                        "age": 150,
                                                        "velocity": [2.5, 1.2],
                                                        "distance_traveled": 45.6,
                                                        "time_in_frame_seconds": 5.0,
                                                        "speed_kmh": 12.5,
                                                        "distance_from_camera_m": 3.2,
                                                    }
                                                },
                                                "summary": {
                                                    "total_tracks": 1,
                                                    "active_tracks": 1,
                                                },
                                            },
                                            "alerts": [
                                                {
                                                    "alert_id": "a1b2c3d4",
                                                    "camera_id": "cam_001",
                                                    "camera_name": "Entrance Camera",
                                                    "timestamp": "2024-01-01T12:00:00",
                                                    "alert_type": "speed",
                                                    "object_class": "person",
                                                    "track_id": "track_42",
                                                    "threshold_value": 10.0,
                                                    "actual_value": 12.5,
                                                    "condition": "over",
                                                    "unit": "km/h",
                                                    "bbox": [100, 200, 300, 400],
                                                    "centroid": [200, 300],
                                                }
                                            ],
                                        },
                                        "calibrated": True,
                                        "frame": "base64_encoded_image_data...",
                                    },
                                },
                                "without_alerts": {
                                    "summary": "Normal message (no alerts)",
                                    "value": {
                                        "camera_id": "cam_001",
                                        "timestamp": 1704106800000,
                                        "results": {
                                            "general_detection": {
                                                "detections": [],
                                                "count": 0,
                                                "model": "general_detection",
                                            }
                                        },
                                        "calibrated": False,
                                    },
                                },
                            },
                        }
                    },
                }
            },
        }
    }

    # Add component schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # Add WebSocket response schema
    openapi_schema["components"]["schemas"]["WebSocketResponse"] = {
        "type": "object",
        "properties": {
            "camera_id": {"type": "string", "description": "Camera identifier"},
            "timestamp": {
                "type": "integer",
                "description": "Unix timestamp in milliseconds",
            },
            "results": {
                "type": "object",
                "description": "Detection, tracking, and alert results",
                "properties": {
                    "{model_name}": {
                        "type": "object",
                        "description": "Detection results from a specific model",
                        "properties": {
                            "detections": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/Detection"},
                            },
                            "count": {"type": "integer"},
                            "model": {"type": "string"},
                        },
                    },
                    "tracking": {
                        "type": "object",
                        "description": "Object tracking results",
                        "properties": {
                            "tracked_objects": {
                                "type": "object",
                                "additionalProperties": {
                                    "$ref": "#/components/schemas/TrackedObject"
                                },
                            },
                            "summary": {
                                "type": "object",
                                "properties": {
                                    "total_tracks": {"type": "integer"},
                                    "active_tracks": {"type": "integer"},
                                },
                            },
                        },
                    },
                    "alerts": {
                        "type": "array",
                        "description": "Triggered alerts (only present when alerts are triggered)",
                        "items": {"$ref": "#/components/schemas/Alert"},
                    },
                },
            },
            "calibrated": {
                "type": "boolean",
                "description": "Whether camera is calibrated for distance/speed",
            },
            "frame": {
                "type": "string",
                "description": "Base64 encoded JPEG image (optional)",
                "nullable": True,
            },
        },
    }

    openapi_schema["components"]["schemas"]["Detection"] = {
        "type": "object",
        "properties": {
            "x1": {"type": "number", "description": "Bounding box top-left X"},
            "y1": {"type": "number", "description": "Bounding box top-left Y"},
            "x2": {"type": "number", "description": "Bounding box bottom-right X"},
            "y2": {"type": "number", "description": "Bounding box bottom-right Y"},
            "confidence": {
                "type": "number",
                "description": "Detection confidence (0-1)",
            },
            "class_id": {"type": "integer", "description": "Class ID"},
            "label": {"type": "string", "description": "Object class name"},
        },
    }

    openapi_schema["components"]["schemas"]["TrackedObject"] = {
        "type": "object",
        "properties": {
            "track_id": {"type": "string", "description": "Unique tracking ID"},
            "class_name": {"type": "string", "description": "Object class"},
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Bounding box [x1, y1, x2, y2]",
            },
            "centroid": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Center point [x, y]",
            },
            "confidence": {"type": "number"},
            "age": {"type": "integer", "description": "Frames tracked"},
            "velocity": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Velocity [vx, vy] in pixels/frame",
            },
            "distance_traveled": {
                "type": "number",
                "description": "Total distance in pixels",
            },
            "time_in_frame_seconds": {
                "type": "number",
                "description": "Time in frame (seconds)",
            },
            "speed_kmh": {
                "type": "number",
                "description": "Speed in km/h (if calibrated)",
                "nullable": True,
            },
            "distance_from_camera_m": {
                "type": "number",
                "description": "Distance from camera in meters (if calibrated)",
                "nullable": True,
            },
        },
    }

    openapi_schema["components"]["schemas"]["Alert"] = {
        "type": "object",
        "description": "Alert triggered by object detection/tracking",
        "properties": {
            "alert_id": {"type": "string", "description": "Unique alert ID"},
            "camera_id": {"type": "string"},
            "camera_name": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "alert_type": {
                "type": "string",
                "enum": ["speed", "tracking", "distance"],
                "description": "Type of alert",
            },
            "object_class": {
                "type": "string",
                "description": "Object that triggered alert",
            },
            "track_id": {"type": "string", "description": "Tracking ID of object"},
            "threshold_value": {
                "type": "number",
                "description": "Configured threshold",
            },
            "actual_value": {"type": "number", "description": "Actual measured value"},
            "condition": {
                "type": "string",
                "enum": ["over", "under"],
                "description": "Alert condition",
            },
            "unit": {
                "type": "string",
                "enum": ["km/h", "seconds", "meters"],
                "description": "Unit of measurement",
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Bounding box of object",
                "nullable": True,
            },
            "centroid": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Center point of object",
                "nullable": True,
            },
        },
    }

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"✅ OpenAPI specification generated: {output_path}")

    # Also generate markdown version
    generate_markdown_docs(openapi_schema, str(output_path).replace(".json", ".md"))

    return openapi_schema


def resolve_schema_ref(ref: str, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a $ref to its actual schema"""
    if not ref.startswith("#/"):
        return {}

    parts = ref.split("/")[1:]  # Remove leading '#'
    schema = openapi_schema

    for part in parts:
        schema = schema.get(part, {})

    return schema


def generate_example_from_schema(
    schema: Dict[str, Any], openapi_schema: Dict[str, Any], depth: int = 0
) -> Any:
    """Generate example data from a JSON schema"""
    if depth > 5:  # Prevent infinite recursion
        return "..."

    # Handle $ref
    if "$ref" in schema:
        resolved = resolve_schema_ref(schema["$ref"], openapi_schema)
        return generate_example_from_schema(resolved, openapi_schema, depth + 1)

    # Handle example if present
    if "example" in schema:
        return schema["example"]

    schema_type = schema.get("type", "object")

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        example = {}
        for prop_name, prop_schema in properties.items():
            # Include if required or if we want to show all properties
            if prop_name in required or depth == 0:
                example[prop_name] = generate_example_from_schema(
                    prop_schema, openapi_schema, depth + 1
                )

        return example

    elif schema_type == "array":
        items = schema.get("items", {})
        item_example = generate_example_from_schema(items, openapi_schema, depth + 1)
        return [item_example]

    elif schema_type == "string":
        if "enum" in schema:
            return schema["enum"][0]
        format_type = schema.get("format", "")
        if format_type == "date-time":
            return "2024-01-01T12:00:00Z"
        elif format_type == "email":
            return "user@example.com"
        elif format_type == "uuid":
            return "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f"
        return schema.get("description", "string")

    elif schema_type == "integer":
        return schema.get("default", 0)

    elif schema_type == "number":
        return schema.get("default", 0.0)

    elif schema_type == "boolean":
        return schema.get("default", True)

    return None


def generate_markdown_docs(openapi_schema: Dict[str, Any], output_file: str):
    """Generate markdown documentation from OpenAPI schema with full examples"""

    md_content = f"""# {openapi_schema.get("info", {}).get("title", "API Documentation")}

Version: {openapi_schema.get("info", {}).get("version", "1.0.0")}

{openapi_schema.get("info", {}).get("description", "")}

---

## Table of Contents

"""

    # Group endpoints by tags
    paths = openapi_schema.get("paths", {})
    endpoints_by_tag = {}

    for path, methods in paths.items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                tags = details.get("tags", ["Untagged"])
                tag = tags[0] if tags else "Untagged"

                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []

                endpoints_by_tag[tag].append(
                    {"path": path, "method": method.upper(), "details": details}
                )

    # Generate TOC
    for tag in sorted(endpoints_by_tag.keys()):
        md_content += f"- [{tag}](#{tag.lower().replace(' ', '-')})\n"

    md_content += "\n---\n\n"

    # Generate endpoint documentation
    for tag in sorted(endpoints_by_tag.keys()):
        md_content += f"## {tag}\n\n"

        for endpoint in endpoints_by_tag[tag]:
            path = endpoint["path"]
            method = endpoint["method"]
            details = endpoint["details"]

            md_content += f"### {method} `{path}`\n\n"
            md_content += f"**Summary:** {details.get('summary', 'No summary')}\n\n"

            if "description" in details:
                md_content += f"{details['description']}\n\n"

            # Parameters
            parameters = details.get("parameters", [])
            if parameters:
                md_content += "**Parameters:**\n\n"
                md_content += "| Name | In | Type | Required | Description |\n"
                md_content += "|------|-------|------|----------|-------------|\n"

                for param in parameters:
                    name = param.get("name", "")
                    in_location = param.get("in", "")
                    param_type = param.get("schema", {}).get("type", "string")
                    required = "✓" if param.get("required", False) else ""
                    description = param.get("description", "")

                    md_content += f"| `{name}` | {in_location} | {param_type} | {required} | {description} |\n"

                md_content += "\n"

            # Request Body
            if "requestBody" in details:
                md_content += "**Request Body:**\n\n"
                content = details["requestBody"].get("content", {})

                if "application/json" in content:
                    schema = content["application/json"].get("schema", {})

                    # Try to get example
                    example = None

                    if "example" in content["application/json"]:
                        example = content["application/json"]["example"]
                    elif "examples" in content["application/json"]:
                        # Get first example
                        examples = content["application/json"]["examples"]
                        if examples:
                            first_example_key = list(examples.keys())[0]
                            example = examples[first_example_key].get("value", {})
                    elif "example" in schema:
                        example = schema["example"]
                    else:
                        # Generate example from schema
                        example = generate_example_from_schema(schema, openapi_schema)

                    if example:
                        md_content += "```json\n"
                        md_content += json.dumps(example, indent=2)
                        md_content += "\n```\n\n"
                    else:
                        md_content += (
                            "```json\n{\n  // Request body structure\n}\n```\n\n"
                        )

            # Responses
            responses = details.get("responses", {})
            if responses:
                md_content += "**Responses:**\n\n"

                for status_code, response in responses.items():
                    md_content += f"**{status_code}** - {response.get('description', 'No description')}\n\n"

                    content = response.get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        examples = content["application/json"].get("examples", {})

                        # Try to get example
                        example = None

                        if examples:
                            for example_name, example_data in examples.items():
                                md_content += f"*{example_data.get('summary', example_name)}:*\n\n"
                                md_content += "```json\n"
                                md_content += json.dumps(
                                    example_data.get("value", {}), indent=2
                                )
                                md_content += "\n```\n\n"
                        elif "example" in content["application/json"]:
                            example = content["application/json"]["example"]
                        elif "example" in schema:
                            example = schema["example"]
                        else:
                            # Generate example from schema
                            example = generate_example_from_schema(
                                schema, openapi_schema
                            )

                        if example and not examples:
                            md_content += "```json\n"
                            md_content += json.dumps(example, indent=2)
                            md_content += "\n```\n\n"

            md_content += "---\n\n"

    # Save markdown
    with open(output_file, "w") as f:
        f.write(md_content)

    print(f"✅ Markdown documentation generated: {output_file}")


if __name__ == "__main__":
    # For standalone usage
    from app.main import app

    generate_openapi_spec(app, "docs/api_documentation.json")
