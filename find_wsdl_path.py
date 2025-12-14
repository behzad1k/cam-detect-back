#!/usr/bin/env python3
# find_wsdl_path.py - Find ONVIF WSDL files on your system

import os
import sys

def find_wsdl_files():
    """Find ONVIF WSDL files and print their location"""
    
    print("🔍 Searching for ONVIF WSDL files...\n")
    
    # Method 1: Check if onvif-zeep is installed
    try:
        import onvif
        onvif_path = os.path.dirname(onvif.__file__)
        wsdl_path = os.path.join(onvif_path, 'wsdl')
        
        print(f"✅ Found onvif-zeep package at: {onvif_path}")
        
        if os.path.exists(wsdl_path):
            print(f"✅ WSDL directory found at: {wsdl_path}")
            
            # List WSDL files
            wsdl_files = [f for f in os.listdir(wsdl_path) if f.endswith('.wsdl')]
            print(f"\n📁 WSDL files found ({len(wsdl_files)}):")
            for f in sorted(wsdl_files):
                print(f"   - {f}")
            
            print(f"\n✅ Use this path in your code:")
            print(f"   WSDL_PATH = '{wsdl_path}'")
            return wsdl_path
        else:
            print(f"❌ WSDL directory NOT found at: {wsdl_path}")
            
    except ImportError:
        print("❌ onvif-zeep package not installed")
        print("   Install with: pip install onvif-zeep")
        return None
    
    # Method 2: Search common locations
    print("\n🔍 Checking common locations...")
    
    common_paths = [
        os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'onvif', 'wsdl'),
        os.path.join(sys.prefix, 'lib', 'python3', 'dist-packages', 'onvif', 'wsdl'),
        '/usr/local/lib/python3.11/site-packages/onvif/wsdl',
        '/usr/local/lib/python3.9/site-packages/onvif/wsdl',
        '/usr/lib/python3/dist-packages/onvif/wsdl',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Found at: {path}")
            return path
        else:
            print(f"❌ Not found: {path}")
    
    print("\n❌ Could not find WSDL files automatically")
    return None


def test_onvif_connection(wsdl_path):
    """Test ONVIF connection with the found WSDL path"""
    print("\n" + "="*80)
    print("🧪 Testing ONVIF Connection")
    print("="*80)
    
    try:
        from onvif import ONVIFCamera
        
        # These are your camera details
        ip = "192.168.1.12"
        port = 80
        username = "admin"
        password = "Behzad8690"
        
        print(f"\n📹 Testing connection to: {ip}:{port}")
        print(f"🔐 Username: {username}")
        print(f"📁 WSDL Path: {wsdl_path}")
        
        # Create camera
        cam = ONVIFCamera(ip, port, username, password, wsdl_path)
        
        print("\n⏳ Connecting...")
        
        # Get device info
        device_info = cam.devicemgmt.GetDeviceInformation()
        
        print("\n✅ CONNECTION SUCCESSFUL!")
        print("\n📊 Camera Information:")
        print(f"   Manufacturer: {device_info.Manufacturer}")
        print(f"   Model: {device_info.Model}")
        print(f"   Firmware: {device_info.FirmwareVersion}")
        print(f"   Serial: {device_info.SerialNumber}")
        
        # Test PTZ
        try:
            print("\n⏳ Testing PTZ capabilities...")
            capabilities = cam.devicemgmt.GetCapabilities()
            
            if hasattr(capabilities, 'PTZ'):
                print("✅ PTZ is supported!")
            else:
                print("⚠️  PTZ capabilities not found")
                
        except Exception as e:
            print(f"⚠️  Could not get PTZ info: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        return False


if __name__ == "__main__":
    print("="*80)
    print("ONVIF WSDL Path Finder & Tester")
    print("="*80 + "\n")
    
    wsdl_path = find_wsdl_files()
    
    if wsdl_path:
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print(f"\n1. Copy this path: {wsdl_path}")
        print("\n2. Update your ptz_control.py:")
        print(f"   WSDL_PATH = '{wsdl_path}'")
        print("\n3. Or use the auto-detection version I provided")
        
        # Ask to test
        print("\n" + "="*80)
        response = input("\nWould you like to test the connection? (y/n): ")
        
        if response.lower() == 'y':
            test_onvif_connection(wsdl_path)
    else:
        print("\n" + "="*80)
        print("TROUBLESHOOTING:")
        print("="*80)
        print("\n1. Make sure onvif-zeep is installed:")
        print("   pip install onvif-zeep")
        print("\n2. Verify installation:")
        print("   python -c 'import onvif; print(onvif.__file__)'")
        print("\n3. If using virtual environment, activate it first")
