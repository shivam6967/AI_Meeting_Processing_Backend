# setup_font.py
import os
import urllib.request

def download_marathi_font():
    font_dir = "fonts"
    font_path = os.path.join(font_dir, "NotoSansDevanagari-Regular.ttf")
    
    # Create fonts directory if it doesn't exist
    os.makedirs(font_dir, exist_ok=True)
    
    if not os.path.exists(font_path):
        print("Downloading Marathi font...")
        try:
            font_url = "https://github.com/notofonts/devanagari/raw/main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf"
            urllib.request.urlretrieve(font_url, font_path)
            print(f"✓ Font downloaded successfully to {font_path}")
            print(f"✓ File size: {os.path.getsize(font_path)} bytes")
        except Exception as e:
            print(f"✗ Error downloading font: {e}")
            return False
    else:
        print(f"✓ Font already exists at {font_path}")
        print(f"✓ File size: {os.path.getsize(font_path)} bytes")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Marathi Font Setup for PDF Generation")
    print("=" * 50)
    success = download_marathi_font()
    if success:
        print("\n✓ Setup completed successfully!")
        print("You can now generate Marathi PDFs.")
    else:
        print("\n✗ Setup failed. Please check your internet connection.")
    print("=" * 50)