# SeeDeep.AI API Documentation

Version: 2.0.0


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
        

---

## Table of Contents

- [PTZ Control](#ptz-control)
- [WebSocket](#websocket)
- [alerts](#alerts)
- [camera-proxy](#camera-proxy)
- [cameras](#cameras)
- [health](#health)
- [onvif](#onvif)

---

## PTZ Control

### POST `/api/v1/ptz/discover`

**Summary:** Discover Camera Endpoint

Discover camera capabilities via ONVIF

Supports multiple authentication methods:
- SOAP with WS-Security
- HTTP Digest Auth
- HTTP Basic Auth

Automatically detects which method works.

**Request Body:**

```json
{
  "ip": "string",
  "username": "string",
  "password": "string"
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/move`

**Summary:** Ptz Move Direct

Control PTZ camera movement using direct IP/credentials

Example:
POST /api/v1/ptz/move?ip=192.168.1.12&username=admin&password=pass&x=0.5&y=0&z=0

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `x` | query | number |  | Pan: -1.0 (left) to 1.0 (right) |
| `y` | query | number |  | Tilt: -1.0 (down) to 1.0 (up) |
| `z` | query | number |  | Zoom: -1.0 (out) to 1.0 (in) |
| `port` | query | integer |  | ONVIF port |
| `timeout` | query | integer |  | Movement timeout in seconds |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/stop`

**Summary:** Ptz Stop Direct

Stop PTZ movement using direct IP/credentials

Example:
POST /api/v1/ptz/stop?ip=192.168.1.12&username=admin&password=pass

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `port` | query | integer |  | ONVIF port |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/ptz/position`

**Summary:** Ptz Get Position Direct

Get current PTZ position using direct IP/credentials

Example:
GET /api/v1/ptz/position?ip=192.168.1.12&username=admin&password=pass

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `port` | query | integer |  | ONVIF port |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/ptz/presets`

**Summary:** Ptz List Presets Direct

List PTZ presets using direct IP/credentials

Example:
GET /api/v1/ptz/presets?ip=192.168.1.12&username=admin&password=pass

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `port` | query | integer |  | ONVIF port |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### DELETE `/api/v1/ptz/presets`

**Summary:** Ptz Remove Preset Direct

Remove a preset

Example:
DELETE /api/v1/ptz/presets?ip=192.168.1.12&username=admin&password=pass&preset_token=1

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `preset_token` | query | string | ✓ | Preset token to remove |
| `port` | query | integer |  | ONVIF port |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/presets/goto`

**Summary:** Ptz Goto Preset Direct

Go to preset using direct IP/credentials

Example:
POST /api/v1/ptz/presets/goto?ip=192.168.1.12&username=admin&password=pass&preset_token=1

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `preset_token` | query | string | ✓ | Preset token to go to |
| `port` | query | integer |  | ONVIF port |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/presets/set`

**Summary:** Ptz Set Preset Direct

Set/create a preset at current position

Example:
POST /api/v1/ptz/presets/set?ip=192.168.1.12&username=admin&password=pass&preset_name=MyPreset

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ | Camera IP address |
| `username` | query | string | ✓ | ONVIF username |
| `password` | query | string | ✓ | ONVIF password |
| `preset_name` | query | string | ✓ | Name for the new preset |
| `preset_token` | query | string |  | Optional preset token to update |
| `port` | query | integer |  | ONVIF port |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/{camera_id}/move`

**Summary:** Ptz Move By Id

Move PTZ camera using camera_id from database

Example:
POST /api/v1/ptz/{camera_id}/move
Body: {"pan": 0.3, "tilt": 0, "zoom": 0, "timeout": 2}

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  // Request body structure
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/{camera_id}/stop`

**Summary:** Ptz Stop By Id

Stop PTZ movement using camera_id

Example:
POST /api/v1/ptz/{camera_id}/stop

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/ptz/{camera_id}/position`

**Summary:** Ptz Get Position By Id

Get PTZ position using camera_id

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/ptz/{camera_id}/presets`

**Summary:** Ptz List Presets By Id

List PTZ presets using camera_id

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/{camera_id}/presets/goto`

**Summary:** Ptz Goto Preset By Id

Go to preset using camera_id

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "preset_token": "string"
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/ptz/{camera_id}/presets/set`

**Summary:** Ptz Set Preset By Id

Set/create preset using camera_id

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "preset_name": "string"
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### DELETE `/api/v1/ptz/{camera_id}/presets/{preset_token}`

**Summary:** Ptz Remove Preset By Id

Remove preset using camera_id

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |
| `preset_token` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/ptz/{camera_id}/capabilities`

**Summary:** Get Ptz Capabilities By Id

Get PTZ capabilities using camera_id

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

## WebSocket

### GET `/ws/camera/{camera_id}`

**Summary:** Camera WebSocket Stream


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
            

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ | Camera ID to stream from |

**Responses:**

**101** - WebSocket connection established

*Message with alerts:*

```json
{
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
          "label": "person"
        }
      ],
      "count": 1,
      "model": "general_detection"
    },
    "tracking": {
      "tracked_objects": {
        "track_42": {
          "track_id": "track_42",
          "class_name": "person",
          "bbox": [
            100,
            200,
            300,
            400
          ],
          "centroid": [
            200,
            300
          ],
          "confidence": 0.95,
          "age": 150,
          "velocity": [
            2.5,
            1.2
          ],
          "distance_traveled": 45.6,
          "time_in_frame_seconds": 5.0,
          "speed_kmh": 12.5,
          "distance_from_camera_m": 3.2
        }
      },
      "summary": {
        "total_tracks": 1,
        "active_tracks": 1
      }
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
        "bbox": [
          100,
          200,
          300,
          400
        ],
        "centroid": [
          200,
          300
        ]
      }
    ]
  },
  "calibrated": true,
  "frame": "base64_encoded_image_data..."
}
```

*Normal message (no alerts):*

```json
{
  "camera_id": "cam_001",
  "timestamp": 1704106800000,
  "results": {
    "general_detection": {
      "detections": [],
      "count": 0,
      "model": "general_detection"
    }
  },
  "calibrated": false
}
```

---

## alerts

### GET `/api/v1/alerts/{camera_id}/config`

**Summary:** Get Alert Config

Get alert configuration for a camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

```json
{
  "cooldown_seconds": 60,
  "distance_alerts": [
    {
      "condition": "under",
      "enabled": true,
      "object_class": "person",
      "threshold_meters": 2.0
    }
  ],
  "email_enabled": true,
  "speed_alerts": [
    {
      "condition": "over",
      "enabled": true,
      "object_class": "car",
      "threshold_kmh": 50.0
    },
    {
      "condition": "over",
      "enabled": true,
      "object_class": "motorcycle",
      "threshold_kmh": 40.0
    }
  ],
  "tracking_alerts": [
    {
      "condition": "over",
      "enabled": true,
      "object_class": "person",
      "threshold_seconds": 30.0
    }
  ]
}
```

**422** - Validation Error

---

### PUT `/api/v1/alerts/{camera_id}/config`

**Summary:** Update Alert Config

Update alert configuration for a camera

Example:
```json
{
  "speed_alerts": [
    {
      "enabled": true,
      "object_class": "car",
      "threshold_kmh": 50.0,
      "condition": "over"
    }
  ],
  "tracking_alerts": [
    {
      "enabled": true,
      "object_class": "person",
      "threshold_seconds": 30.0,
      "condition": "over"
    }
  ],
  "distance_alerts": [
    {
      "enabled": true,
      "object_class": "person",
      "threshold_meters": 2.0,
      "condition": "under"
    }
  ],
  "email_enabled": true,
  "cooldown_seconds": 60
}
```

Note: Alerts will be sent via the existing WebSocket connection for this camera.

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "cooldown_seconds": 60,
  "distance_alerts": [
    {
      "condition": "under",
      "enabled": true,
      "object_class": "person",
      "threshold_meters": 2.0
    }
  ],
  "email_enabled": true,
  "speed_alerts": [
    {
      "condition": "over",
      "enabled": true,
      "object_class": "car",
      "threshold_kmh": 50.0
    },
    {
      "condition": "over",
      "enabled": true,
      "object_class": "motorcycle",
      "threshold_kmh": 40.0
    }
  ],
  "tracking_alerts": [
    {
      "condition": "over",
      "enabled": true,
      "object_class": "person",
      "threshold_seconds": 30.0
    }
  ]
}
```

**Responses:**

**200** - Successful Response

```json
{
  "cooldown_seconds": 60,
  "distance_alerts": [
    {
      "condition": "under",
      "enabled": true,
      "object_class": "person",
      "threshold_meters": 2.0
    }
  ],
  "email_enabled": true,
  "speed_alerts": [
    {
      "condition": "over",
      "enabled": true,
      "object_class": "car",
      "threshold_kmh": 50.0
    },
    {
      "condition": "over",
      "enabled": true,
      "object_class": "motorcycle",
      "threshold_kmh": 40.0
    }
  ],
  "tracking_alerts": [
    {
      "condition": "over",
      "enabled": true,
      "object_class": "person",
      "threshold_seconds": 30.0
    }
  ]
}
```

**422** - Validation Error

---

### PATCH `/api/v1/alerts/{camera_id}/email`

**Summary:** Update Alert Email

Update alert email address for a camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |
| `email` | query | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/alerts/{camera_id}/history`

**Summary:** Get Alert History

Get recent alert history for a camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |
| `limit` | query | integer |  |  |

**Responses:**

**200** - Successful Response

```json
[
  {}
]
```

**422** - Validation Error

---

### GET `/api/v1/alerts/{camera_id}/stats`

**Summary:** Get Alert Stats

Get alert statistics for a camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

```json
{
  "alerts_by_class": {
    "car": 45,
    "person": 82
  },
  "alerts_by_type": {
    "distance": 20,
    "speed": 45,
    "tracking": 62
  },
  "camera_id": "cam_001",
  "last_alert_time": "2024-01-01T12:30:00Z",
  "total_alerts": 127
}
```

**422** - Validation Error

---

### POST `/api/v1/alerts/{camera_id}/test-email`

**Summary:** Test Email Alert

Send test email alert

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

## camera-proxy

### GET `/api/v1/cameras/{camera_id}/stream`

**Summary:** Proxy Camera Stream

Proxy camera MJPEG stream - browser compatible version

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

## cameras

### POST `/api/v1/cameras/`

**Summary:** Create Camera

Create a new camera

**Request Body:**

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_email": "alerts@company.com",
  "features": {
    "counting": false,
    "detection": true,
    "detection_classes": [
      "person",
      "car",
      "motorcycle"
    ],
    "distance": false,
    "distance_classes": [],
    "speed": true,
    "speed_classes": [
      "car",
      "motorcycle"
    ],
    "tracking": true,
    "tracking_classes": [
      "person"
    ]
  },
  "fps": 30,
  "height": 1080,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "width": 1920
}
```

**Responses:**

**201** - Successful Response

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_config": {
    "cooldown_seconds": 60,
    "distance_alerts": [],
    "email_enabled": true,
    "speed_alerts": [],
    "tracking_alerts": []
  },
  "alert_email": "alerts@company.com",
  "calibration_mode": "reference_object",
  "calibration_points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "features": {
    "detection": true,
    "detection_classes": [
      "person",
      "car"
    ],
    "distance": true,
    "speed": true,
    "tracking": true
  },
  "fps": 30,
  "height": 1080,
  "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
  "ip_address": "192.168.1.12",
  "is_active": true,
  "is_calibrated": true,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "onvif_port": 80,
  "pixels_per_meter": 125.5,
  "rtsp_port": 554,
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "updated_at": "2024-01-01T13:00:00Z",
  "username": "admin",
  "width": 1920
}
```

**422** - Validation Error

---

### GET `/api/v1/cameras/`

**Summary:** List Cameras

List all cameras

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `active_only` | query | boolean |  |  |

**Responses:**

**200** - Successful Response

```json
[
  {
    "active_models": [
      "general_detection"
    ],
    "alert_config": {
      "cooldown_seconds": 60,
      "distance_alerts": [],
      "email_enabled": true,
      "speed_alerts": [],
      "tracking_alerts": []
    },
    "alert_email": "alerts@company.com",
    "calibration_mode": "reference_object",
    "calibration_points": [
      {
        "pixel_x": 100,
        "pixel_y": 200,
        "real_x": 0,
        "real_y": 0
      },
      {
        "pixel_x": 350,
        "pixel_y": 200,
        "real_x": 2,
        "real_y": 0
      }
    ],
    "created_at": "2024-01-01T12:00:00Z",
    "features": {
      "detection": true,
      "detection_classes": [
        "person",
        "car"
      ],
      "distance": true,
      "speed": true,
      "tracking": true
    },
    "fps": 30,
    "height": 1080,
    "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
    "ip_address": "192.168.1.12",
    "is_active": true,
    "is_calibrated": true,
    "location": "Main Entrance",
    "name": "Entrance Camera",
    "onvif_port": 80,
    "pixels_per_meter": 125.5,
    "rtsp_port": 554,
    "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
    "updated_at": "2024-01-01T13:00:00Z",
    "username": "admin",
    "width": 1920
  }
]
```

**422** - Validation Error

---

### GET `/api/v1/cameras/{camera_id}`

**Summary:** Get Camera

Get a specific camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_config": {
    "cooldown_seconds": 60,
    "distance_alerts": [],
    "email_enabled": true,
    "speed_alerts": [],
    "tracking_alerts": []
  },
  "alert_email": "alerts@company.com",
  "calibration_mode": "reference_object",
  "calibration_points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "features": {
    "detection": true,
    "detection_classes": [
      "person",
      "car"
    ],
    "distance": true,
    "speed": true,
    "tracking": true
  },
  "fps": 30,
  "height": 1080,
  "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
  "ip_address": "192.168.1.12",
  "is_active": true,
  "is_calibrated": true,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "onvif_port": 80,
  "pixels_per_meter": 125.5,
  "rtsp_port": 554,
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "updated_at": "2024-01-01T13:00:00Z",
  "username": "admin",
  "width": 1920
}
```

**422** - Validation Error

---

### PATCH `/api/v1/cameras/{camera_id}`

**Summary:** Update Camera

Update a camera - FIXED with retry logic for concurrent access

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "features": {
    "detection": true,
    "speed": true,
    "tracking": true
  },
  "is_active": true,
  "location": "New Location",
  "name": "Updated Camera Name"
}
```

**Responses:**

**200** - Successful Response

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_config": {
    "cooldown_seconds": 60,
    "distance_alerts": [],
    "email_enabled": true,
    "speed_alerts": [],
    "tracking_alerts": []
  },
  "alert_email": "alerts@company.com",
  "calibration_mode": "reference_object",
  "calibration_points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "features": {
    "detection": true,
    "detection_classes": [
      "person",
      "car"
    ],
    "distance": true,
    "speed": true,
    "tracking": true
  },
  "fps": 30,
  "height": 1080,
  "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
  "ip_address": "192.168.1.12",
  "is_active": true,
  "is_calibrated": true,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "onvif_port": 80,
  "pixels_per_meter": 125.5,
  "rtsp_port": 554,
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "updated_at": "2024-01-01T13:00:00Z",
  "username": "admin",
  "width": 1920
}
```

**422** - Validation Error

---

### DELETE `/api/v1/cameras/{camera_id}`

**Summary:** Delete Camera

Delete a camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**204** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/cameras/{camera_id}/calibrate`

**Summary:** Calibrate Camera

Calibrate a camera for pixel-to-meter conversion

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "mode": "reference_object",
  "points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "reference_width_meters": 2.0
}
```

**Responses:**

**200** - Successful Response

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_config": {
    "cooldown_seconds": 60,
    "distance_alerts": [],
    "email_enabled": true,
    "speed_alerts": [],
    "tracking_alerts": []
  },
  "alert_email": "alerts@company.com",
  "calibration_mode": "reference_object",
  "calibration_points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "features": {
    "detection": true,
    "detection_classes": [
      "person",
      "car"
    ],
    "distance": true,
    "speed": true,
    "tracking": true
  },
  "fps": 30,
  "height": 1080,
  "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
  "ip_address": "192.168.1.12",
  "is_active": true,
  "is_calibrated": true,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "onvif_port": 80,
  "pixels_per_meter": 125.5,
  "rtsp_port": 554,
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "updated_at": "2024-01-01T13:00:00Z",
  "username": "admin",
  "width": 1920
}
```

**422** - Validation Error

---

### GET `/api/v1/cameras/{camera_id}/models`

**Summary:** Get Available Models

Get available detection models

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### PATCH `/api/v1/cameras/{camera_id}/features`

**Summary:** Update Camera Features

Update camera feature configuration - FIXED with retry logic

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "counting": false,
  "detection": true,
  "detection_classes": [
    "person",
    "car",
    "motorcycle"
  ],
  "distance": true,
  "distance_classes": [
    "person"
  ],
  "speed": true,
  "speed_classes": [
    "car",
    "motorcycle"
  ],
  "tracking": true,
  "tracking_classes": [
    "person",
    "car"
  ]
}
```

**Responses:**

**200** - Successful Response

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_config": {
    "cooldown_seconds": 60,
    "distance_alerts": [],
    "email_enabled": true,
    "speed_alerts": [],
    "tracking_alerts": []
  },
  "alert_email": "alerts@company.com",
  "calibration_mode": "reference_object",
  "calibration_points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "features": {
    "detection": true,
    "detection_classes": [
      "person",
      "car"
    ],
    "distance": true,
    "speed": true,
    "tracking": true
  },
  "fps": 30,
  "height": 1080,
  "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
  "ip_address": "192.168.1.12",
  "is_active": true,
  "is_calibrated": true,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "onvif_port": 80,
  "pixels_per_meter": 125.5,
  "rtsp_port": 554,
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "updated_at": "2024-01-01T13:00:00Z",
  "username": "admin",
  "width": 1920
}
```

**422** - Validation Error

---

### POST `/api/v1/cameras/test-connection`

**Summary:** Test Camera Connection

Test camera connection (RTSP/HTTP/WEBCAM) and return a preview frame

**Request Body:**

```json
{
  "rtsp_url": "string"
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### PATCH `/api/v1/cameras/{camera_id}/detection-classes`

**Summary:** Update Detection Classes

Update detection classes for a camera

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
[
  "string"
]
```

**Responses:**

**200** - Successful Response

```json
{
  "active_models": [
    "general_detection"
  ],
  "alert_config": {
    "cooldown_seconds": 60,
    "distance_alerts": [],
    "email_enabled": true,
    "speed_alerts": [],
    "tracking_alerts": []
  },
  "alert_email": "alerts@company.com",
  "calibration_mode": "reference_object",
  "calibration_points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "features": {
    "detection": true,
    "detection_classes": [
      "person",
      "car"
    ],
    "distance": true,
    "speed": true,
    "tracking": true
  },
  "fps": 30,
  "height": 1080,
  "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
  "ip_address": "192.168.1.12",
  "is_active": true,
  "is_calibrated": true,
  "location": "Main Entrance",
  "name": "Entrance Camera",
  "onvif_port": 80,
  "pixels_per_meter": 125.5,
  "rtsp_port": 554,
  "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
  "updated_at": "2024-01-01T13:00:00Z",
  "username": "admin",
  "width": 1920
}
```

**422** - Validation Error

---

### GET `/api/v1/cameras/{camera_id}/calibration`

**Summary:** Get Calibration Info

Get current calibration information

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### DELETE `/api/v1/cameras/{camera_id}/calibration`

**Summary:** Clear Calibration

Clear camera calibration

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/cameras/{camera_id}/calibration/test`

**Summary:** Test Calibration

Test calibration without saving (preview mode)

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Request Body:**

```json
{
  "mode": "reference_object",
  "points": [
    {
      "pixel_x": 100,
      "pixel_y": 200,
      "real_x": 0,
      "real_y": 0
    },
    {
      "pixel_x": 350,
      "pixel_y": 200,
      "real_x": 2,
      "real_y": 0
    }
  ],
  "reference_width_meters": 2.0
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### GET `/api/v1/cameras/{camera_id}/frame`

**Summary:** Get Camera Frame

Get a single frame from camera (RTSP/HTTP/WEBCAM) for calibration

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `camera_id` | path | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

## health

### GET `/`

**Summary:** Root

Root endpoint

**Responses:**

**200** - Successful Response

---

### GET `/health`

**Summary:** Health Check

Health check endpoint

**Responses:**

**200** - Successful Response

---

## onvif

### POST `/api/v1/onvif/discover`

**Summary:** Discover Camera

Discover camera capabilities via ONVIF

Fixed for Dahua cameras with proper WS-Security authentication

**Request Body:**

```json
{
  "ip": "string",
  "username": "string",
  "password": "string"
}
```

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/onvif/ptz/move`

**Summary:** Ptz Move

Control PTZ camera movement

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ |  |
| `username` | query | string | ✓ |  |
| `password` | query | string | ✓ |  |
| `x` | query | number |  |  |
| `y` | query | number |  |  |
| `z` | query | number |  |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

### POST `/api/v1/onvif/ptz/stop`

**Summary:** Ptz Stop

Stop PTZ movement

**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| `ip` | query | string | ✓ |  |
| `username` | query | string | ✓ |  |
| `password` | query | string | ✓ |  |

**Responses:**

**200** - Successful Response

**422** - Validation Error

---

