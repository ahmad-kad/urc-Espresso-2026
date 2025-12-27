#!/usr/bin/env python3
"""
Extract and display per-class accuracy from benchmark results
"""

import json
import sys

def main():
    try:
        with open('output/benchmark_results.json', 'r') as f:
            data = json.load(f)

        print('=' * 100)
        print('COMPREHENSIVE MODEL BENCHMARK RESULTS WITH PER-CLASS ACCURACY')
        print('=' * 100)
        print()

        # Filter successful PyTorch results
        pytorch_results = [r for r in data if r.get('status') == 'success' and r.get('type') == 'pytorch']

        print(f'Total PyTorch models evaluated: {len(pytorch_results)}')
        print()

        for result in pytorch_results:
            print(f'MODEL: {result["name"]}')
            print(f'  Format: {result.get("format", "N/A")}')
            print(f'  Size: {result.get("size_mb", "N/A"):.1f} MB')
            print(f'  FPS: {result.get("fps", "N/A"):.1f}')
            print(f'  Inference Time: {result.get("avg_ms", "N/A"):.1f} ms')
            print(f'  Overall mAP50: {result.get("mAP50", "N/A"):.3f}')
            print(f'  Overall Precision: {result.get("precision", "N/A"):.3f}')
            print(f'  Overall Recall: {result.get("recall", "N/A"):.3f}')
            print(f'  Overall F1-Score: {result.get("f1_score", "N/A"):.3f}')

            # Check for per-class metrics
            if 'per_class' in result and result['per_class']:
                print('  PER-CLASS ACCURACY BREAKDOWN:')
                per_class = result['per_class']
                print('    Class Name        | Precision | Recall | mAP50 | F1-Score')
                print('    ------------------|-----------|--------|-------|----------')

                for class_name, metrics in per_class.items():
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    map50 = metrics.get('mAP50', 0)
                    f1 = metrics.get('f1', 0)

                    print('15'
                          '9.3f'
                          '7.3f'
                          '7.3f'
                          '9.3f')
            else:
                print('  PER-CLASS ACCURACY: Not available (evaluation may have failed)')

            print()
            print('-' * 80)
            print()

        # Summary table
        print('=' * 100)
        print('SUMMARY TABLE - ALL MODELS')
        print('=' * 100)

        print('Model Name              | Size(MB) | FPS  | mAP50 | Prec. | Rec. | F1    | Status')
        print('------------------------|----------|------|-------|-------|------|-------|--------')

        for result in pytorch_results:
            name = result.get('name', 'Unknown')[:22]  # Truncate long names
            size_mb = result.get('size_mb', 0)
            fps = result.get('fps', 0)
            map50 = result.get('mAP50', 0)
            precision = result.get('precision', 0)
            recall = result.get('recall', 0)
            f1 = result.get('f1_score', 0)
            status = 'OK' if result.get('status') == 'success' else 'FAIL'

            print('22'
                  '8.1f'
                  '6.1f'
                  '7.3f'
                  '7.3f'
                  '6.3f'
                  '7.3f'
                  '8')

        # Best models analysis
        print()
        print('=' * 100)
        print('BEST MODELS BY CATEGORY')
        print('=' * 100)

        # Best accuracy
        if pytorch_results:
            best_accuracy = max(pytorch_results, key=lambda x: x.get('mAP50', 0))
            print(f'BEST ACCURACY: {best_accuracy["name"]} - {best_accuracy.get("mAP50", 0):.3f} mAP50')

            # Best speed
            best_speed = max(pytorch_results, key=lambda x: x.get('fps', 0))
            print(f'BEST SPEED: {best_speed["name"]} - {best_speed.get("fps", 0):.1f} FPS')

            # Best memory efficiency
            best_memory = min(pytorch_results, key=lambda x: x.get('size_mb', float('inf')))
            print(f'BEST MEMORY: {best_memory["name"]} - {best_memory.get("size_mb", 0):.1f} MB')

            # Best balance (high accuracy, reasonable size)
            balanced = sorted(pytorch_results,
                            key=lambda x: (x.get('mAP50', 0) * 0.7 - x.get('size_mb', 0) * 0.3),
                            reverse=True)[0]
            print(f'BEST BALANCE: {balanced["name"]} - {balanced.get("mAP50", 0):.3f} mAP50, {balanced.get("size_mb", 0):.1f} MB')

    except Exception as e:
        print(f"Error reading benchmark results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
