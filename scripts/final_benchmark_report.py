#!/usr/bin/env python3
"""
Generate final comprehensive benchmark report
"""

import pandas as pd

def main():
    print("=" * 120)
    print("COMPREHENSIVE MODEL BENCHMARK REPORT - CONFIDENCE & MEMORY FOCUS")
    print("=" * 120)

    # Load data
    df = pd.read_csv('output/benchmark_summary.csv')
    df = df[df['status'] == 'success'].copy()
    df['mAP50'] = pd.to_numeric(df['mAP50'], errors='coerce')
    df['fps'] = pd.to_numeric(df['fps'], errors='coerce')
    df['size_mb'] = pd.to_numeric(df['size_mb'], errors='coerce')

    # Separate by type
    pytorch_df = df[df['type'] == 'pytorch'].copy()

    print(f"\nTOTAL MODELS BENCHMARKED: {len(df)} (PyTorch: {len(pytorch_df)}, ONNX: {len(df) - len(pytorch_df)})")

    # CONFIDENCE OPTIMIZED MODELS
    print(f"\n{'CONFIDENCE OPTIMIZED MODELS':^80}")
    print("=" * 80)
    confidence_df = pytorch_df[pytorch_df['name'].str.contains('confidence', case=False)]
    if not confidence_df.empty:
        confidence_sorted = confidence_df.sort_values('size_mb')
        print("Model Name                    | Size(MB) | FPS  | mAP50 | Precision | Recall | F1")
        print("------------------------------|----------|------|-------|-----------|--------|-----")
        for _, row in confidence_sorted.iterrows():
            name = str(row['name'])[:28]
            print("28"
                  "10.1f"
                  "6.1f"
                  "7.3f"
                  "11.3f"
                  "8.3f"
                  "5.3f")

    # LOW MEMORY MODELS
    print(f"\n{'ULTRA-LOW MEMORY MODELS (< 10MB)':^80}")
    print("=" * 80)
    small_df = pytorch_df[pytorch_df['size_mb'] < 10].sort_values('size_mb')
    if not small_df.empty:
        print("Model Name                    | Size(MB) | FPS  | mAP50 | Precision | Recall | F1")
        print("------------------------------|----------|------|-------|-----------|--------|-----")
        for _, row in small_df.iterrows():
            name = str(row['name'])[:28]
            print("28"
                  "10.1f"
                  "6.1f"
                  "7.3f"
                  "11.3f"
                  "8.3f"
                  "5.3f")

    # ACCURACY LEADERS
    print(f"\n{'ACCURACY LEADERS (mAP50)':^80}")
    print("=" * 80)
    if not pytorch_df.empty and 'mAP50' in pytorch_df.columns:
        accuracy_sorted = pytorch_df.sort_values('mAP50', ascending=False).head(8)
        print("Model Name                    | Size(MB) | FPS  | mAP50 | Precision | Recall | F1")
        print("------------------------------|----------|------|-------|-----------|--------|-----")
        for _, row in accuracy_sorted.iterrows():
            name = str(row['name'])[:28]
            print("28"
                  "10.1f"
                  "6.1f"
                  "7.3f"
                  "11.3f"
                  "8.3f"
                  "5.3f")

    # SPEED CHAMPIONS
    print(f"\n{'SPEED CHAMPIONS':^80}")
    print("=" * 80)
    speed_df = df.dropna(subset=['fps']).sort_values('fps', ascending=False).head(8)
    if not speed_df.empty:
        print("Model Name                    | Size(MB) | FPS  | mAP50 | Format")
        print("------------------------------|----------|------|-------|--------")
        for _, row in speed_df.iterrows():
            name = str(row['name'])[:28]
            map50 = row.get('mAP50', 'N/A')
            if pd.notna(map50):
                map50_str = ".3f"
            else:
                map50_str = "N/A"
            print("28"
                  "10.1f"
                  "6.1f"
                  "7"
                  "8")

    # STATISTICS
    print(f"\n{'STATISTICAL SUMMARY':^80}")
    print("=" * 80)
    if not pytorch_df.empty:
        print(f"PyTorch Models ({len(pytorch_df)} total):")
        print(".3f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".3f")

    if len(df) > len(pytorch_df):
        onnx_df = df[df['type'].isin(['onnx', 'int8'])]
        if not onnx_df.empty:
            print(f"\nONNX Models ({len(onnx_df)} total):")
            print(".1f")
            print(".1f")

    # RECOMMENDATIONS
    print(f"\n{'FINAL RECOMMENDATIONS':^80}")
    print("=" * 80)

    print("Based on your priorities (CONFIDENCE + LOW MEMORY FOOTPRINT):")

    # Best overall
    if not confidence_df.empty:
        best_overall = confidence_df.sort_values(['size_mb', 'mAP50'], ascending=[True, False]).head(1)
        if not best_overall.empty:
            row = best_overall.iloc[0]
            print("BEST OVERALL CHOICE:")
            print("1f"
                  ".3f"
                  ".1f"
                  ".3f")

    # Best lightweight
    if not small_df.empty:
        best_light = small_df.sort_values('mAP50', ascending=False).head(1)
        if not best_light.empty:
            row = best_light.iloc[0]
            print("\nBEST ULTRA-LIGHTWEIGHT:")
            print("1f"
                  ".3f"
                  ".1f"
                  ".3f")

    # Best accuracy
    if not pytorch_df.empty:
        best_acc = pytorch_df.sort_values('mAP50', ascending=False).head(1)
        if not best_acc.empty:
            row = best_acc.iloc[0]
            print("\nBEST ACCURACY:")
            print("1f"
                  ".3f"
                  ".1f"
                  ".3f")

    print(f"\nDEPLOYMENT SUMMARY")
    print("-" * 25)
    print("PyTorch models: Full accuracy metrics, ready for production")
    print("ONNX models: Cross-platform deployment with speed improvements")
    print("INT8 models: Compatibility issues with complex architectures")
    print("All results saved in output/ directory")

    print(f"\n{'=' * 120}")

if __name__ == "__main__":
    main()



