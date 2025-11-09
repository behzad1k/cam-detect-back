#!/usr/bin/env python3
"""
Quick script to check and fix camera configurations
Run this to enable detections on all cameras
"""

import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

# Database URL - update if different
DATABASE_URL = "sqlite+aiosqlite:///./seedeep.db"


async def check_and_fix_cameras():
  """Check and fix camera configurations"""

  print("🔍 Connecting to database...")
  engine = create_async_engine(DATABASE_URL, echo=False)

  async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
  )

  async with async_session() as session:
    # Get all cameras
    result = await session.execute("""SELECT * FROM cameras""")
    cameras = result.fetchall()

    if not cameras:
      print("❌ No cameras found in database")
      return

    print(f"\n📹 Found {len(cameras)} camera(s)\n")

    for camera in cameras:
      camera_id = camera[0]  # id
      name = camera[1]  # name
      active_models = camera[12]  # active_models
      features = camera[11]  # features
      fps = camera[7]  # fps

      print(f"Camera: {name} ({camera_id})")
      print(f"  FPS: {fps}")
      print(f"  Active Models: {active_models}")
      print(f"  Features: {features}")

      # Check issues
      issues = []

      if not active_models or active_models == '[]':
        issues.append("No active models")

      if not features or '"detection": true' not in str(features):
        issues.append("Detection not enabled")

      if fps and fps > 20:
        issues.append(f"High FPS ({fps})")

      if issues:
        print(f"  ⚠️  Issues: {', '.join(issues)}")

        # Fix it
        fix_query = """
                UPDATE cameras 
                SET 
                    active_models = '["face_detection", "general_detection"]',
                    features = '{"detection": true, "tracking": false, "speed": false, "distance": false}',
                    fps = 15
                WHERE id = :camera_id
                """

        await session.execute(fix_query, {"camera_id": camera_id})
        print(f"  ✅ Fixed!")
      else:
        print(f"  ✅ Configuration OK")

      print()

    await session.commit()
    print("💾 Changes saved to database")

  await engine.dispose()


async def show_available_models():
  """Show available AI models"""
  print("\n📦 Available Models:")
  print("  - face_detection (Facemask.pt)")
  print("  - cap_detection (Cap.pt)")
  print("  - weapon_detection (Weapon.pt)")
  print("  - fire_detection (Fire.pt)")
  print("  - general_detection (YOLO.pt)")
  print("\n💡 Make sure model files exist in app/models/weights/\n")


if __name__ == "__main__":
  print("=" * 60)
  print("  Camera Configuration Checker & Fixer")
  print("=" * 60)

  try:
    asyncio.run(check_and_fix_cameras())
    asyncio.run(show_available_models())

    print("\n✅ Done! Restart your backend server to apply changes.")
    print("   python main.py\n")

  except Exception as e:
    print(f"\n❌ Error: {e}")
    print("   Make sure:")
    print("   1. Database file exists (seedeep.db)")
    print("   2. You're in the project root directory")
    print("   3. Required packages are installed: pip install sqlalchemy aiosqlite")
    sys.exit(1)