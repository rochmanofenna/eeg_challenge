#!/usr/bin/env python3
"""
HBN Dataset Explorer
Lists all available HBN EEG datasets and their contents
"""

import s3fs
from collections import defaultdict
import sys


def explore_hbn_datasets():
    """Explore all available HBN datasets"""

    print("=== HBN EEG Dataset Explorer ===")
    print("Connecting to: s3://fcp-indi/data/Projects/HBN/BIDS_EEG/")

    try:
        fs = s3fs.S3FileSystem(anon=True)
        bucket_path = "fcp-indi/data/Projects/HBN/BIDS_EEG"

        # Get all releases
        contents = fs.ls(bucket_path, detail=False)
        releases = [item.split('/')[-1] for item in contents if 'cmi_bids' in item]
        releases = sorted(releases)

        print(f"\nFound {len(releases)} HBN releases:")

        total_subjects = 0
        total_files = 0
        dataset_summary = []

        for release in releases:
            print(f"\n--- {release} ---")

            try:
                # Get subjects
                release_path = f"{bucket_path}/{release}"
                subjects = fs.ls(release_path, detail=False)
                subject_names = [s.split('/')[-1] for s in subjects if 'sub-' in s]

                print(f"  Subjects: {len(subject_names)}")

                # Sample EEG files from first subject
                if subject_names:
                    first_subject = subject_names[0]
                    try:
                        eeg_path = f"{release_path}/{first_subject}/eeg"
                        eeg_files = fs.ls(eeg_path, detail=False)

                        # Count different task types
                        tasks = defaultdict(int)
                        for file_path in eeg_files:
                            filename = file_path.split('/')[-1]
                            if '_task-' in filename and filename.endswith('.set'):
                                task = filename.split('_task-')[1].split('_')[0]
                                tasks[task] += 1

                        print(f"  Tasks available: {list(tasks.keys())}")
                        print(f"  Sample subject: {first_subject}")

                        # Estimate total files
                        files_per_subject = len([f for f in eeg_files if f.endswith('.set')])
                        estimated_total_files = files_per_subject * len(subject_names)

                        total_subjects += len(subject_names)
                        total_files += estimated_total_files

                        dataset_summary.append({
                            'release': release,
                            'subjects': len(subject_names),
                            'tasks': list(tasks.keys()),
                            'files_per_subject': files_per_subject,
                            'estimated_total_files': estimated_total_files
                        })

                    except Exception as e:
                        print(f"  Could not access EEG files: {e}")

            except Exception as e:
                print(f"  Error accessing {release}: {e}")

        # Print summary
        print(f"\n=== Dataset Summary ===")
        print(f"Total releases: {len(releases)}")
        print(f"Total subjects: {total_subjects}")
        print(f"Estimated total EEG files: {total_files}")

        print(f"\n=== Available Tasks Across All Releases ===")
        all_tasks = set()
        for summary in dataset_summary:
            all_tasks.update(summary['tasks'])

        for task in sorted(all_tasks):
            releases_with_task = [s['release'] for s in dataset_summary if task in s['tasks']]
            print(f"  {task}: Available in {len(releases_with_task)} releases")

        print(f"\n=== Recommended Training Configurations ===")

        print(f"\n1. Quick Test (Small dataset):")
        print(f"   --releases cmi_bids_R1 --max-subjects 10")
        print(f"   Expected: ~10 subjects, ~50-100 EEG files")

        print(f"\n2. Medium Training (Balanced):")
        print(f"   --releases cmi_bids_R1 cmi_bids_R2 cmi_bids_R3 --max-subjects 50")
        print(f"   Expected: ~150 subjects, ~750+ EEG files")

        print(f"\n3. Large Training (All data):")
        large_releases = [r for r in releases if 'R' in r and 'NC' not in r][:5]
        print(f"   --releases {' '.join(large_releases)} --max-subjects 100")
        print(f"   Expected: ~500 subjects, ~2500+ EEG files")

        print(f"\n4. Full Dataset (Maximum):")
        print(f"   --releases {' '.join(releases)}")
        print(f"   Expected: {total_subjects} subjects, ~{total_files} EEG files")

        return dataset_summary

    except ImportError:
        print("Error: s3fs not installed. Run: pip install s3fs")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main function"""
    dataset_summary = explore_hbn_datasets()

    if dataset_summary:
        print(f"\n=== Ready to Train! ===")
        print(f"Use train_hbn_complete.py with desired configuration:")
        print(f"\nExamples:")
        print(f"python train_hbn_complete.py --releases cmi_bids_R1 --max-subjects 10 --test-only")
        print(f"python train_hbn_complete.py --releases cmi_bids_R1 cmi_bids_R2 --max-subjects 20")

        # Save summary
        import json
        with open('hbn_dataset_summary.json', 'w') as f:
            json.dump(dataset_summary, f, indent=2)
        print(f"\nDataset summary saved to: hbn_dataset_summary.json")


if __name__ == "__main__":
    main()