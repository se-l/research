root = r'D:\trade\data\option\usa'

def delete_iv_files(root: str, dry_run: bool = True) -> int:
    """
    Recursively delete files whose filenames contain 'iv_quote' or 'iv_trade'.
    Prints each matched path. Returns count of matched files.
    """
    import os

    patterns = ("iv_quote", "iv_trade")
    count = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            lower = fname.lower()
            if any(p in lower for p in patterns):
                full_path = os.path.join(dirpath, fname)
                print(("[DRY RUN] " if dry_run else "") + full_path)
                count += 1
                if not dry_run:
                    try:
                        os.remove(full_path)
                    except FileNotFoundError:
                        # File may have been removed concurrently
                        pass
                    except PermissionError as e:
                        print(f"PermissionError: {full_path} -> {e}")
                    except OSError as e:
                        print(f"OSError: {full_path} -> {e}")
    return count


if __name__ == '__main__':
    # Dry run first
    matched = delete_iv_files(root, dry_run=False)
    print(f"Matched files (dry run): {matched}")

    # Uncomment to perform actual deletion
    # matched = delete_iv_files(root, dry_run=False)
    # print(f"Deleted files: {matched}")

