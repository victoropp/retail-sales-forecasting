"""Debug script to find the exact categorical error"""
import sys
sys.path.insert(0, 'src')
import traceback

try:
    import data_loader
    df = data_loader.load_and_merge_all()
    print("SUCCESS!")
except Exception as e:
    print("="*80)
    print("ERROR DETAILS:")
    print("="*80)
    traceback.print_exc()
    print("="*80)
