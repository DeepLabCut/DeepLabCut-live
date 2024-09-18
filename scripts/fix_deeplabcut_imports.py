"""Script to update DeepLabCut imports when copying predictors"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RecursiveImportFixer:
    """Recursively fixes imports in python files"""

    import_prefix: str
    new_import_prefix: str
    dry_run: bool = False

    def fix_imports(self, target: Path) -> None:
        if target.is_dir():
            self._walk_folder(target)
        elif target.suffix == ".py":
            self._fix_imports(target)
        else:
            raise ValueError(f"Oops! You can only fix `.py` files (not {target})")

    def _walk_folder(self, folder: Path) -> None:
        if not folder.is_dir():
            raise ValueError(f"Oops! Something went wrong (not a folder): {folder}")

        for file in folder.iterdir():
            if file.suffix == ".py":
                self._fix_imports(file)
            elif file.is_dir():
                self._walk_folder(file)

    def _fix_imports(self, file: Path) -> None:
        if not file.suffix == ".py":
            raise ValueError(f"Oops! Something went wrong: {file}")

        print(f"Fixing file {file}")
        with open(file, "r") as f:
            file_content = f.readlines()

        fixed_lines = []
        for index, line in enumerate(file_content):
            parsed = line
            if self.import_prefix in line:
                parsed = line.replace(self.import_prefix, self.new_import_prefix)
                print(f"  Found import on line {index}")
                print(f"    original: ```{line}```")
                print(f"    fixed:    ```{parsed}```")

            fixed_lines.append(parsed)

        if not self.dry_run:
            with open(file, "w") as f:
                f.writelines(fixed_lines)


def main(
    target: Path,
    import_prefix: str,
    new_import_prefix: str,
    dry_run: bool,
) -> None:
    print(
        f"Replacing all imports of {import_prefix}.* in {target} with an import of "
        f"{new_import_prefix}.*"
    )
    fixer = RecursiveImportFixer(import_prefix, new_import_prefix, dry_run=dry_run)
    fixer.fix_imports(target)


if __name__ == "__main__":
    main(
        target=Path("../dlclive/models").resolve(),
        import_prefix="deeplabcut.pose_estimation_pytorch.models",
        new_import_prefix="dlclive.models",
        dry_run=True,
    )
    main(
        target=Path("../dlclive/models").resolve(),
        import_prefix="deeplabcut.pose_estimation_pytorch.registry",
        new_import_prefix="dlclive.models.registry",
        dry_run=True,
    )
