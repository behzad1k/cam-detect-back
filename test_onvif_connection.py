#!/usr/bin/env python3
"""
Test python-onvif package vs onvif-zeep
"""

import sys


def test_python_onvif():
    """Test with python-onvif package"""
    print("=" * 80)
    print("TESTING: python-onvif")
    print("=" * 80)

    try:
        from onvif import ONVIFCamera

        print("✅ python-onvif is installed")

        # Camera details
        ip = "192.168.1.13"
        port = 80
        username = "admin"
        password = "Behzad8690"

        print(f"\n📹 Camera: {ip}:{port}")
        print(f"👤 Username: {username}")

        print("\n⏳ Creating camera connection...")

        # python-onvif uses slightly different initialization
        cam = ONVIFCamera(ip, port, username, password)

        print("✅ Camera object created")

        print("\n⏳ Getting device information...")
        device_info = cam.devicemgmt.GetDeviceInformation()

        print("\n✅ SUCCESS WITH python-onvif!")
        print(f"\n📊 Device Information:")
        print(f"   Manufacturer: {device_info.Manufacturer}")
        print(f"   Model: {device_info.Model}")
        print(f"   Firmware: {device_info.FirmwareVersion}")

        # Try getting media profiles
        print("\n⏳ Getting media profiles...")
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()

        print(f"✅ Found {len(profiles)} profile(s)")

        for i, profile in enumerate(profiles, 1):
            print(f"\n   Profile {i}: {profile.Name}")

            # Get stream URI
            try:
                stream_uri = media_service.GetStreamUri(
                    {
                        "StreamSetup": {
                            "Stream": "RTP-Unicast",
                            "Transport": {"Protocol": "RTSP"},
                        },
                        "ProfileToken": profile.token,
                    }
                )
                print(f"   URI: {stream_uri.Uri}")
            except Exception as e:
                print(f"   ⚠️ Could not get URI: {e}")

        # Check PTZ
        print("\n⏳ Checking PTZ capabilities...")
        try:
            ptz_service = cam.create_ptz_service()
            print("✅ PTZ is supported")
        except:
            print("ℹ️  PTZ not available")

        print("\n" + "=" * 80)
        print("🎉 python-onvif WORKS!")
        print("=" * 80)

        print("\n✅ Recommendation: Use python-onvif")
        print("   Install: pip install python-onvif")
        print("   Uninstall old: pip uninstall onvif-zeep")

        return True

    except ImportError:
        print("❌ python-onvif is NOT installed")
        print("\nInstall it with:")
        print("   pip install python-onvif")
        return False

    except Exception as e:
        print(f"\n❌ python-onvif FAILED: {e}")
        print("\nError details:")
        import traceback

        traceback.print_exc()
        return False


def test_onvif_zeep():
    """Test with onvif-zeep package"""
    print("\n" + "=" * 80)
    print("TESTING: onvif-zeep")
    print("=" * 80)

    try:
        # First check which package is actually imported
        import onvif

        print(f"ℹ️  onvif package location: {onvif.__file__}")

        from onvif import ONVIFCamera

        # Camera details
        ip = "192.168.1.13"
        port = 80
        username = "admin"
        password = "Behzad8690"

        print(f"\n📹 Camera: {ip}:{port}")

        # Find WSDL path
        import os

        onvif_dir = os.path.dirname(onvif.__file__)
        wsdl_path = os.path.join(onvif_dir, "wsdl")

        if not os.path.exists(wsdl_path):
            # Try alternative location
            from pathlib import Path

            site_packages = Path(sys.prefix) / "lib"
            for wsdl_dir in site_packages.rglob("wsdl"):
                devicemgmt = wsdl_dir / "devicemgmt.wsdl"
                if devicemgmt.exists():
                    wsdl_path = str(wsdl_dir)
                    break

        print(f"📁 WSDL: {wsdl_path}")

        print("\n⏳ Creating camera connection...")
        cam = ONVIFCamera(ip, port, username, password, wsdl_path)

        print("✅ Camera object created")

        print("\n⏳ Getting device information...")
        device_info = cam.devicemgmt.GetDeviceInformation()

        print("\n✅ SUCCESS WITH onvif-zeep!")
        print(f"\n📊 Device Information:")
        print(f"   Manufacturer: {device_info.Manufacturer}")
        print(f"   Model: {device_info.Model}")

        print("\n" + "=" * 80)
        print("🎉 onvif-zeep WORKS!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ onvif-zeep FAILED: {e}")
        return False


def compare_packages():
    """Compare both packages"""
    print("=" * 80)
    print("ONVIF PACKAGE COMPARISON")
    print("=" * 80)

    print("\nPackage Details:")
    print("\n1. onvif-zeep:")
    print("   - Actively maintained fork")
    print("   - Uses zeep for SOAP")
    print("   - Requires WSDL files")
    print("   - Can have WS-Security issues")

    print("\n2. python-onvif:")
    print("   - Original package")
    print("   - Simpler implementation")
    print("   - Better compatibility with some cameras")
    print("   - May be less actively maintained")

    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    # Test python-onvif first
    result_python_onvif = test_python_onvif()

    # Test onvif-zeep
    result_onvif_zeep = test_onvif_zeep()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(f"\npython-onvif:  {'✅ WORKS' if result_python_onvif else '❌ FAILED'}")
    print(f"onvif-zeep:    {'✅ WORKS' if result_onvif_zeep else '❌ FAILED'}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if result_python_onvif and not result_onvif_zeep:
        print("\n✅ Use python-onvif")
        print("\nSteps:")
        print("   1. pip uninstall onvif-zeep")
        print("   2. pip install python-onvif")
        print("   3. Update code to use python-onvif")

    elif result_onvif_zeep and not result_python_onvif:
        print("\n✅ Stick with onvif-zeep")
        print("   (Already working with your setup)")

    elif result_python_onvif and result_onvif_zeep:
        print("\n✅ Both work! Choose based on preference:")
        print("   - python-onvif: Simpler, fewer dependencies")
        print("   - onvif-zeep: More actively maintained")

    else:
        print("\n❌ Neither package works")
        print("   Stick with manual SOAP implementation (onvif_universal.py)")


if __name__ == "__main__":
    compare_packages()
