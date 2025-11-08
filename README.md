# SeeDeep.AI - Multi-Camera Object Detection & Tracking System

A scalable, real-time object detection and tracking system supporting multiple simultaneous camera streams with YOLO models, camera calibration, speed detection, and analytics.

## 🚀 Features

- **Multi-Camera Support**: Process multiple camera streams simultaneously
- **YOLO Detection**: Support for multiple YOLO models (PPE, Fire, Weapon, Face, etc.)
- **Object Tracking**: Real-time object tracking with unique IDs
- **Camera Calibration**: Pixel-to-meter conversion for real-world measurements
- **Speed Detection**: Calculate object speeds in pixels/second and meters/second
- **Zone Analytics**: Define zones and track object occupancy
- **Database Persistence**: Store camera configurations and calibration data
- **WebSocket Streaming**: Real-time bidirectional communication
- **REST API**: Complete CRUD operations for camera management

## 📋 Requirements

- Python 3.11+
- PostgreSQL (or SQLite for development)
- CUDA-capable GPU (optional, but recommended)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd seedeep-ai
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup database

```bash
# Update DATABASE_URL in .env file
# Then run migrations
alembic upgrade head
```

### 5. Add YOLO model weights

Place your YOLO model files in `models/weights/`:
- `PPE.pt`
- `Fire.pt`
- `Weapon.pt`
- `Facemask.pt`
- etc.

### 6. Run the application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Camera Management

#### Create Camera

```bash
POST /api/v1/cameras
Content-Type: application/json

{
  "name": "Main Entrance",
  "location": "Building A - Floor 1",
  "rtsp_url": "rtsp://camera-ip:554/stream",
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "features": {
    "detection": true,
    "tracking": true,
    "speed": true,
    "counting": false
  },
  "active_models": ["ppe_detection", "fire_detection"]
}
```

#### List Cameras

```bash
GET /api/v1/cameras
GET /api/v1/cameras?active_only=true
```

#### Get Camera

```bash
GET /api/v1/cameras/{camera_id}
```

#### Update Camera

```bash
PATCH /api/v1/cameras/{camera_id}
Content-Type: application/json

{
  "features": {
    "detection": true,
    "tracking": true,
    "speed": true
  },
  "active_models": ["ppe_detection"]
}
```

#### Delete Camera

```bash
DELETE /api/v1/cameras/{camera_id}
```

### Camera Calibration

#### Calibrate Camera

```bash
POST /api/v1/cameras/{camera_id}/calibrate
Content-Type: application/json

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
      "pixel_x": 500,
      "pixel_y": 200,
      "real_x": 2.0,
      "real_y": 0
    }
  ],
  "reference_width_meters": 2.0
}
```

This calculates the pixels-per-meter ratio for real-world measurements.

## 🔌 WebSocket Protocol

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Binary Frame Format

Send frames in the following binary format:

```
[camera_id_length (1 byte)]
[camera_id (variable bytes, UTF-8)]
[timestamp (4 bytes, little-endian)]
[image_data (JPEG/PNG)]
```

### Example Client (JavaScript)

```javascript
class CameraClient {
  constructor(cameraId) {
    this.cameraId = cameraId;
    this.ws = new WebSocket('ws://localhost:8000/ws');
    
    this.ws.onopen = () => console.log('Connected');
    this.ws.onmessage = (event) => this.handleMessage(event);
    this.ws.onerror = (error) => console.error('WebSocket error:', error);
  }
  
  async sendFrame(imageBlob) {
    const timestamp = Math.floor(Date.now() / 1000);
    const cameraIdBytes = new TextEncoder().encode(this.cameraId);
    
    // Calculate total size
    const headerSize = 1 + cameraIdBytes.length + 4;
    const imageBuffer = await imageBlob.arrayBuffer();
    const totalSize = headerSize + imageBuffer.byteLength;
    
    // Build binary message
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const uint8View = new Uint8Array(buffer);
    
    let offset = 0;
    
    // Camera ID length
    view.setUint8(offset, cameraIdBytes.length);
    offset += 1;
    
    // Camera ID
    uint8View.set(cameraIdBytes, offset);
    offset += cameraIdBytes.length;
    
    // Timestamp
    view.setUint32(offset, timestamp, true); // little-endian
    offset += 4;
    
    // Image data
    uint8View.set(new Uint8Array(imageBuffer), offset);
    
    // Send
    this.ws.send(buffer);
  }
  
  handleMessage(event) {
    const data = JSON.parse(event.data);
    
    if (data.error) {
      console.error('Error:', data.error);
      return;
    }
    
    console.log('Results:', data);
    
    // Handle detection results
    if (data.results) {
      for (const [modelName, modelResult] of Object.entries(data.results)) {
        console.log(`${modelName}: ${modelResult.count} detections`);
      }
    }
    
    // Handle tracking results
    if (data.results.tracking) {
      const tracking = data.results.tracking;
      console.log(`Tracked objects: ${tracking.summary.total_tracks}`);
      
      for (const [trackId, obj] of Object.entries(tracking.tracked_objects)) {
        console.log(`Track ${trackId}: ${obj.class_name} at speed ${obj.speed_m_per_sec || obj.speed_px_per_sec} m/s`);
      }
    }
  }
}

// Usage
const client = new CameraClient('camera-uuid-here');

// From video element
videoElement.addEventListener('play', () => {
  setInterval(async () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    canvas.toBlob((blob) => {
      client.sendFrame(blob);
    }, 'image/jpeg', 0.8);
  }, 100); // Send frame every 100ms
});
```

### Response Format

#### Detection Response

```json
{
  "camera_id": "abc123",
  "timestamp": 1699123456789,
  "calibrated": true,
  "results": {
    "ppe_detection": {
      "detections": [
        {
          "x1": 100,
          "y1": 150,
          "x2": 200,
          "y2": 300,
          "confidence": 0.92,
          "class_id": 0,
          "label": "helmet"
        }
      ],
      "count": 1,
      "model": "ppe_detection",
      "error": null
    }
  }
}
```

#### Tracking Response

```json
{
  "camera_id": "abc123",
  "timestamp": 1699123456789,
  "calibrated": true,
  "results": {
    "tracking": {
      "tracked_objects": {
        "a1b2c3d4": {
          "track_id": "a1b2c3d4",
          "class_name": "person",
          "bbox": [100, 150, 200, 300],
          "centroid": [150, 225],
          "confidence": 0.92,
          "age": 45,
          "velocity": [2.5, -1.2],
          "distance_traveled": 125.5,
          "speed_px_per_sec": 15.2,
          "speed_m_per_sec": 0.76,
          "speed_kmh": 2.74
        }
      },
      "summary": {
        "total_tracks": 3,
        "active_tracks": 2
      }
    }
  }
}
```

## 🗂️ Database Schema

### Camera Table

```sql
CREATE TABLE cameras (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    location VARCHAR,
    rtsp_url VARCHAR,
    width INTEGER DEFAULT 640,
    height INTEGER DEFAULT 480,
    fps INTEGER DEFAULT 30,
    is_calibrated BOOLEAN DEFAULT FALSE,
    pixels_per_meter FLOAT,
    calibration_mode VARCHAR,
    calibration_points JSON,
    features JSON,
    active_models JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│  (Multiple)     │
└────────┬────────┘
         │ WebSocket
         ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   WebSocket Handler          │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │   Stream Manager             │  │
│  │  (Multi-camera support)      │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │   Detection Service          │  │
│  │   - YOLO Models              │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │   Tracking Service           │  │
│  │   - Object Tracking          │  │
│  │   - Speed Calculation        │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │   Camera Service             │  │
│  │   - CRUD Operations          │  │
│  │   - Calibration              │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │   PostgreSQL Database        │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

## 📈 Scaling Considerations

1. **Multiple Cameras**: The system supports unlimited simultaneous camera streams
2. **Load Balancing**: Deploy multiple instances behind a load balancer
3. **Database**: Use PostgreSQL with connection pooling
4. **Redis**: Add Redis for caching and pub/sub (optional)
5. **GPU**: Use CUDA-enabled GPUs for faster inference
6. **Microservices**: Split detection/tracking into separate services if needed

## 🔧 Configuration

Edit `.env` file:

```bash
# Server
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/db

# Models
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
MAX_DETECTIONS=100
FORCE_CPU=false

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.