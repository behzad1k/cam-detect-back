#!/usr/bin/env python3
import onvif
import os
import sys

print("onvif package:", onvif.__file__)
onvif_dir = os.path.dirname(onvif.__file__)
print("onvif directory:", onvif_dir)

wsdl_path = os.path.join(onvif_dir, 'wsdl')
print("Expected WSDL path:", wsdl_path)
print("Exists:", os.path.exists(wsdl_path))

if os.path.exists(wsdl_path):
    print("\nWSDL files:")
    for f in os.listdir(wsdl_path):
        if f.endswith('.wsdl'):
            print(f"  - {f}")
else:
    print("\n❌ WSDL directory not found!")
    print("\nSearching for wsdl directory...")
    
    # Search in parent directories
    search_dirs = [
        onvif_dir,
        os.path.join(onvif_dir, '..'),
        os.path.join(sys.prefix, 'lib'),
        os.path.join(sys.prefix, 'share'),
    ]
    
    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if 'wsdl' in dirs:
                    wsdl_candidate = os.path.join(root, 'wsdl')
                    # Check if it contains ONVIF WSDL files
                    try:
                        wsdl_files = os.listdir(wsdl_candidate)
                        if any('devicemgmt' in f for f in wsdl_files):
                            print(f"\n✅ Found ONVIF WSDL at: {wsdl_candidate}")
                    except:
                        pass
