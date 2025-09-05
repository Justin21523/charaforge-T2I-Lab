#!/usr/bin/env python3
"""
SD UI Lab - Release Management Script
Automates version bumping, tagging, and release preparation
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse


class ReleaseManager:
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.version_files = {
            "pyproject.toml": r'version = "([^"]+)"',
            "setup.py": r'version=["\']([^"\']+)["\']',
            "shared_cache_init.py": r'__version__ = ["\']([^"\']+)["\']',
        }

    def get_current_version(self) -> Optional[str]:
        """Get current version from git tags"""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                return result.stdout.strip().lstrip("v")
            return None
        except Exception:
            return None

    def bump_version(self, current: str, bump_type: str) -> str:
        """Bump version according to semantic versioning"""
        major, minor, patch = map(int, current.split("."))

        if bump_type == "major":
            return f"{major + 1}.0.0"
        elif bump_type == "minor":
            return f"{major}.{minor + 1}.0"
        elif bump_type == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

    def update_version_files(self, new_version: str) -> List[str]:
        """Update version in all relevant files"""
        updated_files = []

        for file_path, pattern in self.version_files.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, "r") as f:
                        content = f.read()

                    new_content = re.sub(pattern, f'version = "{new_version}"', content)

                    if new_content != content:
                        with open(full_path, "w") as f:
                            f.write(new_content)
                        updated_files.append(file_path)
                        print(f"✅ Updated {file_path}")
                except Exception as e:
                    print(f"⚠️  Failed to update {file_path}: {e}")

        return updated_files

    def generate_changelog(self, since_version: str = None) -> str:
        """Generate changelog from git commits"""
        since_arg = f"v{since_version}..HEAD" if since_version else "HEAD"

        try:
            result = subprocess.run(
                ["git", "log", since_arg, "--pretty=format:%h %s (%an)", "--no-merges"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                return "No changes found."

            commits = result.stdout.strip().split("\n")
            if not commits or commits == [""]:
                return "No changes found."

            # Categorize commits by conventional commit types
            categories = {
                "feat": "✨ Features",
                "fix": "🐛 Bug Fixes",
                "docs": "📚 Documentation",
                "style": "💄 Style Changes",
                "refactor": "♻️ Code Refactoring",
                "perf": "⚡ Performance Improvements",
                "test": "✅ Tests",
                "chore": "🔧 Chores",
                "build": "👷 Build System",
                "ci": "💚 CI/CD",
            }

            categorized = {cat: [] for cat in categories.values()}
            categorized["🔄 Other Changes"] = []

            for commit in commits:
                # Parse conventional commit format
                match = re.match(r"^(\w+) (.+?) \((.+?)\)$", commit)
                if match:
                    commit_hash, message, author = match.groups()

                    # Extract type and scope
                    type_match = re.match(r"^(\w+)(?:\([^)]+\))?: (.+)$", message)
                    if type_match:
                        commit_type, clean_message = type_match.groups()
                        category = categories.get(commit_type, "🔄 Other Changes")
                        categorized[category].append(
                            f"- {clean_message} ({commit_hash})"
                        )
                    else:
                        categorized["🔄 Other Changes"].append(
                            f"- {message} ({commit_hash})"
                        )

            # Build changelog
            changelog_parts = []
            for category_name in categories.values():
                if categorized[category_name]:
                    changelog_parts.append(f"\n### {category_name}\n")
                    changelog_parts.extend(categorized[category_name])

            if categorized["🔄 Other Changes"]:
                changelog_parts.append(f"\n### 🔄 Other Changes\n")
                changelog_parts.extend(categorized["🔄 Other Changes"])

            return (
                "\n".join(changelog_parts)
                if changelog_parts
                else "No categorized changes found."
            )

        except Exception as e:
            return f"Error generating changelog: {e}"

    def run_tests(self) -> bool:
        """Run test suite before release"""
        print("🧪 Running test suite...")

        test_script = self.project_root / "tests" / "test_smoke.py"
        if not test_script.exists():
            print("⚠️  No test suite found, skipping tests")
            return True

        try:
            result = subprocess.run(
                [sys.executable, str(test_script)], cwd=self.project_root
            )

            success = result.returncode == 0
            print(
                f"{'✅' if success else '❌'} Tests {'passed' if success else 'failed'}"
            )
            return success

        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False

    def check_git_status(self) -> Tuple[bool, List[str]]:
        """Check if working directory is clean"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                return False, ["Git status check failed"]

            untracked = []
            modified = []

            for line in result.stdout.strip().split("\n"):
                if line:
                    status = line[:2]
                    filename = line[3:]

                    if "?" in status:
                        untracked.append(filename)
                    elif any(c in status for c in "MAD"):
                        modified.append(filename)

            issues = []
            if modified:
                issues.append(f"Modified files: {', '.join(modified)}")
            if untracked:
                issues.append(f"Untracked files: {', '.join(untracked)}")

            return len(issues) == 0, issues

        except Exception as e:
            return False, [f"Git status error: {e}"]

    def create_git_tag(self, version: str, changelog: str) -> bool:
        """Create annotated git tag"""
        tag_name = f"v{version}"

        # Create tag message
        tag_message = f"Release {tag_name}\n\n{changelog}"

        try:
            # Create annotated tag
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", tag_message],
                check=True,
                cwd=self.project_root,
            )

            print(f"✅ Created tag {tag_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create tag: {e}")
            return False

    def create_release(
        self, bump_type: str, dry_run: bool = False, skip_tests: bool = False
    ) -> bool:
        """Create a new release"""
        print(f"🚀 Creating {bump_type} release...")

        # Get current version
        current_version = self.get_current_version()
        if not current_version:
            print("⚠️  No current version found, starting with 1.0.0")
            current_version = "0.0.0"

        new_version = self.bump_version(current_version, bump_type)
        print(f"📈 Version: {current_version} → {new_version}")

        if dry_run:
            print("🔍 DRY RUN - No changes will be made")

        # Check git status
        is_clean, issues = self.check_git_status()
        if not is_clean and not dry_run:
            print("❌ Working directory is not clean:")
            for issue in issues:
                print(f"   {issue}")
            print("Commit or stash changes before releasing.")
            return False

        # Run tests
        if not skip_tests and not dry_run:
            if not self.run_tests():
                print("❌ Tests failed, aborting release")
                return False

        # Generate changelog
        changelog = self.generate_changelog(current_version)
        print(f"\n📝 Changelog:\n{changelog}\n")

        if dry_run:
            print("🔍 DRY RUN - Would update these files:")
            for file_path in self.version_files:
                if (self.project_root / file_path).exists():
                    print(f"   - {file_path}")
            print(f"🔍 DRY RUN - Would create tag: v{new_version}")
            return True

        # Update version files
        updated_files = self.update_version_files(new_version)
        if not updated_files:
            print("⚠️  No version files were updated")

        # Commit version changes
        if updated_files:
            try:
                subprocess.run(
                    ["git", "add"] + updated_files, check=True, cwd=self.project_root
                )
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"chore(release): bump version to {new_version}",
                    ],
                    check=True,
                    cwd=self.project_root,
                )
                print("✅ Committed version changes")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to commit changes: {e}")
                return False

        # Create git tag
        if not self.create_git_tag(new_version, changelog):
            return False

        print(f"\n🎉 Release v{new_version} created successfully!")
        print("\nNext steps:")
        print(f"1. Push changes: git push origin main")
        print(f"2. Push tags: git push origin v{new_version}")
        print("3. Create GitHub release with changelog")

        return True


def main():
    parser = argparse.ArgumentParser(description="SD UI Lab Release Manager")
    parser.add_argument(
        "bump_type", choices=["major", "minor", "patch"], help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running tests before release"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (default: parent of script dir)",
    )

    args = parser.parse_args()

    manager = ReleaseManager(args.project_root)
    success = manager.create_release(args.bump_type, args.dry_run, args.skip_tests)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
