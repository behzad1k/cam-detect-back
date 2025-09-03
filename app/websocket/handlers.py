
# Add these imports at the top of your file if not already present:
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from io import BytesIO
import numpy as np
# Quick fix for your existing WebSocket handler
# Replace your current WebSocket endpoint with this:

async def websocket_endpoint(websocket: WebSocket):
  await websocket.accept()
  print("✅ WebSocket connected")

  try:
    while True:
      # CRITICAL FIX: Use receive() instead of receive_text()
      message = await websocket.receive()

      # Check message type
      if "text" in message:
        # Handle JSON messages (control commands)
        print(f"📝 Received text message: {message['text'][:100]}...")
        await handle_text_message(websocket, message["text"])

      elif "bytes" in message:
        # Handle binary messages (image frames)
        print(f"📸 Received binary message: {len(message['bytes'])} bytes")
        await handle_binary_message(websocket, message["bytes"])

      else:
        print(f"❓ Unknown message type: {list(message.keys())}")

  except WebSocketDisconnect:
    print("🔌 WebSocket disconnected")
  except Exception as e:
    print(f"❌ WebSocket error: {e}")
    await websocket.send_text(json.dumps({
      "type": "error",
      "message": str(e)
    }))


async def handle_text_message(websocket: WebSocket, text_data: str):
  """Handle JSON control messages"""
  try:
    data = json.loads(text_data)
    message_type = data.get("type")

    print(f"🎛️ Handling control message: {message_type}")

    if message_type == "configure_tracking":
      print("⚙️ Configuring tracking...")
      await websocket.send_text(json.dumps({
        "type": "tracking_configured",
        "success": True
      }))

    elif message_type == "start_tracking":
      print("▶️ Starting tracking...")
      await websocket.send_text(json.dumps({
        "type": "tracking_started"
      }))

    elif message_type == "stop_tracking":
      print("⏹️ Stopping tracking...")
      await websocket.send_text(json.dumps({
        "type": "tracking_stopped"
      }))

    else:
      print(f"❓ Unknown message type: {message_type}")

  except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")


async def handle_binary_message(websocket: WebSocket, binary_data: bytes):
  """Handle binary image data"""
  try:
    print(f"🖼️ Processing image frame: {len(binary_data)} bytes")

    # Parse your binary format
    offset = 0

    # Read timestamp (4 bytes)
    timestamp = int.from_bytes(binary_data[offset:offset + 4], 'little')
    offset += 4
    print(f"⏰ Timestamp: {timestamp}")

    # Read model count (1 byte)
    model_count = binary_data[offset]
    offset += 1
    print(f"🧠 Model count: {model_count}")

    # Parse models
    models = []
    for i in range(model_count):
      # Model name length
      name_length = binary_data[offset]
      offset += 1

      # Model name
      model_name = binary_data[offset:offset + name_length].decode('utf-8')
      offset += name_length

      # Class filter flag
      has_filter = binary_data[offset]
      offset += 1

      class_filter = []
      if has_filter:
        class_count = binary_data[offset]
        offset += 1

        for j in range(class_count):
          class_name_length = binary_data[offset]
          offset += 1
          class_name = binary_data[offset:offset + class_name_length].decode('utf-8')
          offset += class_name_length
          class_filter.append(class_name)

      models.append({
        'name': model_name,
        'class_filter': class_filter if class_filter else None
      })

    print(f"🔍 Models to use: {[m['name'] for m in models]}")

    # Image data (remaining bytes)
    image_data = binary_data[offset:]
    print(f"📷 Image data: {len(image_data)} bytes")

    # Process with your existing model manager
    detections = await process_with_your_models(image_data, models)

    # Send results back
    await websocket.send_text(json.dumps({
      "type": "detections",  # or "tracking_results" if tracking enabled
      "results": {
        "detections": detections,
        "count": len(detections)
      },
      "timestamp": timestamp
    }))

    print(f"✅ Sent {len(detections)} detections back to client")

  except Exception as e:
    print(f"❌ Error processing binary data: {e}")
    await websocket.send_text(json.dumps({
      "type": "error",
      "message": f"Image processing error: {str(e)}"
    }))


async def process_with_your_models(image_data: bytes, models: list):
  """Integrate with your existing model processing"""
  try:
    # Convert bytes to PIL Image
    from PIL import Image
    from io import BytesIO
    import numpy as np

    image_io = BytesIO(image_data)
    image = Image.open(image_io)
    image_array = np.array(image)

    print(f"🖼️ Image loaded: {image.size}, mode: {image.mode}")

    # TODO: Replace this with your actual model manager integration
    # Example:
    from app.models.model_manager import model_manager
    all_detections = []
    for model_config in models:
        detections = model_manager.process_detections(
            model_config['name'],
            image_array,
            model_config['class_filter']
        )
        all_detections.extend(detections)
    return all_detections

  except Exception as e:
    print(f"❌ Model processing error: {e}")
    return []
