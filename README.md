# SeeDeep.Ai - Real-time Object Detection API

A FastAPI-based real-time object detection service that processes video frames via WebSocket connections using YOLO models.

## Features

- **Real-time Processing**: WebSocket-based frame processing
- **Multiple Models**: Support for multiple YOLO models (face detection, cap detection, PPE detection)
- **Class Filtering**: Users can specify which classes they want to detect from each model
- **Modular Architecture**: Clean, maintainable code structure
- **GPU Support**: Automatic GPU detection and usage
- **Health Monitoring**: Built-in health check and model status endpoints

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app setup
│   ├── core/
│   │   ├── config.py          # Configuration settings
│   │   └── schemas.py         # Pydantic models
│   ├── models/
│   │   └── model_manager.py   # Model loading and inference
│   ├── websocket/
│   │   ├── connection_manager.py  # WebSocket connection handling
│   │   └── handlers.py        # WebSocket message handlers
│   ├── api/
│   │   └── routes.py          # REST API endpoints
│   └── utils/
│       ├── frame_processor.py # Frame processing utilities
│       └── logging.py         # Logging configuration
├── main.py                    # Application entry point
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your YOLO model files in the `weights/` directory:
   - `weights/Facemask.pt`
   - `weights/Cap.pt`
   - `weights/PPE.pt`

## Usage

### Starting the Server

```bash
python main.py
```

### Environment Variables

- `DEBUG`: Enable debug mode (default: false)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.5)
- `MAX_DETECTIONS`: Maximum detections per frame (default: 100)
- `FORCE_CPU`: Force CPU usage (default: false)

### API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /models`: List all models and their status
- `GET /models/{model_name}/classes`: Get available classes for a model
- `POST /models/{model_name}/load`: Load a specific model
- `POST /models/{model_name}/unload`: Unload a specific model
- `WebSocket /ws`: Real-time frame processing

### WebSocket Protocol

The WebSocket endpoint expects binary data with the following structure:

1. **Timestamp** (4 bytes): Frame timestamp
2. **Model Count** (1 byte): Number of models to use
3. **For each model**:
   - Model name length (1 byte)
   - Model name (variable bytes)
   - Has class filter flag (1 byte): 0 or 1
   - If has class filter:
     - Class count (1 byte)
     - For each class:
       - Class name length (1 byte)
       - Class name (variable bytes)
4. **Image Data** (remaining bytes): JPEG/PNG encoded image

### Class Filtering

You can specify which classes you want to detect from each model by including class filters in your WebSocket message. This allows you to:

- Only detect specific objects (e.g., only "person" and "car" from a general detection model)
- Reduce processing time by filtering unwanted detections
- Customize detection results per use case

### Example Client Code

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Prepare frame data with class filtering
function sendFrame(imageData, models) {
    const timestamp = Math.floor(Date.now() / 1000);
    const buffer = new ArrayBuffer(/* calculate size */);
    const view = new DataView(buffer);
    
    let offset = 0;
    
    // Timestamp
    view.setUint32(offset, timestamp);
    offset += 4;
    
    // Model count
    view.setUint8(offset, models.length);
    offset += 1;
    
    // For each model
    models.forEach(model => {
        // Model name
        const nameBytes = new TextEncoder().encode(model.name);
        view.setUint8(offset, nameBytes.length);
        offset += 1;
        // ... copy name bytes
        
        // Class filter
        if (model.classFilter) {
            view.setUint8(offset, 1); // has filter
            offset += 1;
            view.setUint8(offset, model.classFilter.length);
            offset += 1;
            // ... encode class names
        } else {
            view.setUint8(offset, 0); // no filter
            offset += 1;
        }
    });
    
    // Image data
    // ... append image bytes
    
    ws.send(buffer);
}

// Example usage
sendFrame(imageData, [
    {
        name: "ppe_detection",
        classFilter: ["helmet", "vest"] // Only detect helmets and vests
    },
    {
        name: "face_detection"
        // No filter - detect all classes
    }
]);
```

## Response Format

```json
{
    "type": "detections",
    "results": {
        "ppe_detection": {
            "detections": [
                {
                    "x1": 100,
                    "y1": 100,
                    "x2": 200,
                    "y2": 200,
                    "confidence": 0.85,
                    "class_id": 0,
                    "label": "helmet"
                }
            ],
            "count": 1,
            "model": "ppe_detection",
            "error": null
        }
    },
    "timestamp": 1634567890000
}
```
