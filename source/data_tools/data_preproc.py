from pathlib import Path
from typing  import Iterator, Tuple, List, Dict
import fnmatch, os


class RecordingIndex:
    """
    Fast lookup / iterator for IKEA recordings.
    This class builds a 3-level index of all recorder directories
    """

    # ── interaction ↔ recorder tables (immutable tuples) ──────────────────
    REC_TYPES_ARIA = (
        ("gripper", "aria_gripper"),
        ("gripper", "aria_human"),
        ("hand",    "aria_human"),
        ("spot",    "aria_spot"),
        ("umi",     "aria_human"),
        ("wrist",   "aria_human"),
        ("wrist",   "aria_wrist"),
    )
    REC_TYPES_IPHONE = (
        ("gripper", "iphone_*"),   # wildcard → iphone_1, iphone_2, …
        ("hand",    "iphone_*"),
        ("spot",    "iphone_*"),
        ("umi",     "iphone_*"),
        ("wrist",   "iphone_*"),
    )
    REC_TYPES_SPOT    = (("spot",  "spot"),)
    REC_TYPES_UMI     = (("umi",   "umi_gripper"),)
    REC_TYPES_GRIPPER = (("gripper","gripper"),)

    ALL_REC_TYPES = (
        REC_TYPES_ARIA
        + REC_TYPES_IPHONE
        + REC_TYPES_SPOT
        + REC_TYPES_UMI
        + REC_TYPES_GRIPPER
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._index: Dict[str, Dict[str, Dict[str, Path]]] | None = None

    # ── build a 3-level dictionary once ───────────────────────────────────
    def _build(self) -> None:
        """
        self._index[location][interaction][recorder] = Path(<recorder_dir>)
        """
        idx: Dict[str, Dict[str, Dict[str, Path]]] = {}

        for loc_dir in self.root.iterdir():
            if not loc_dir.is_dir():
                continue
            loc = loc_dir.name
            for int_dir in loc_dir.iterdir():
                if not int_dir.is_dir():
                    continue
                interaction = int_dir.name
                for rec_dir in int_dir.iterdir():
                    if not rec_dir.is_dir():
                        continue
                    recorder = rec_dir.name

                    # accept only if (interaction, recorder) matches a table entry
                    if not any(
                        interaction == pat_int and fnmatch.fnmatch(recorder, pat_rec)
                        for pat_int, pat_rec in self.ALL_REC_TYPES
                    ):
                        continue

                    idx \
                      .setdefault(loc, {}) \
                      .setdefault(interaction, {})[recorder] = rec_dir

        self._index = idx

    # ── public helpers -----------------------------------------------------
    @property
    def index(self):                       # lazy-build
        if self._index is None:
            self._build()
        return self._index

    @property
    def locations(self) -> List[str]:
        return sorted(self.index.keys())

    def iter_recorders(
        self
    ) -> Iterator[Tuple[str, str, str, Path]]:
        """
        Yield (location, interaction, recorder, recorder_dir)
        for every recorder directory that exists.
        """
        for loc, int_dict in self.index.items():
            for interaction, rec_dict in int_dict.items():
                for recorder, path in rec_dict.items():
                    yield loc, interaction, recorder, path

    # quick helper: list all *.vrs files under one recorder dir -------------
    @staticmethod
    def _is_vrs(f: str) -> bool:
        return f.endswith(".vrs") and not f.startswith("._")

    def vrs_under_recorder(self, recorder_dir: Path) -> List[Path]:
        files: List[Path] = []
        for p, _, fnames in os.walk(recorder_dir):
            for f in fnames:
                if self._is_vrs(f):
                    files.append(Path(p) / f)
        return files
    
    @staticmethod
    def _parse_interaction_index(name: str) -> str | None:
        """
        Extract <idx> from anything like
            bedroom_1_1-8_gripper.vrs
            bedroom_1_1-8_gripper          (dir)
            bedroom_1_1-2-4_umi.MP4

        Works even when the <location> part itself contains underscores.
        """
        import re
        m = re.match(
            r"""^
                .+?_                 #  greedy prefix up to **last** '_' before idx
                (?P<idx>\d+(?:-\d+)*)_  #  1 or more numbers joined by '-'
                [^_]+                #  next underscore + suffix (interaction)
                (?:\..+)?$           #  optional file extension
            """,
            name,
            re.VERBOSE,
        )
        return m.group("idx") if m else None

    # ── flexible query (now with idx) ----------------------------------------
    def query(
        self,
        location:          str | None = None,
        interaction:       str | None = None,
        recorder:          str | None = None,
        interaction_index: str | None = None,   # NEW
    ):
        """
        Flexible lookup.

        Parameters
        ----------
        location, interaction, recorder
            Same as before.  Wild-cards allowed in `recorder` (fnmatch rules).
            If recorder string contains no '*' or '?', it's treated as a prefix
            and expanded to "<recorder>*" so  recorder="aria"  → "aria*".
        interaction_index
            *None*  → behave exactly as before (return recorder dirs).
            str     → filter recordings whose middle token matches this pattern
                       (wild-cards allowed).  Returns one tuple per *recording*
                       (loc, inter, recorder, idx, rec_path).

        Returns
        -------
        list of tuples
            • if interaction_index is None:
                (loc, interaction, recorder, recorder_dir)
            • else:
                (loc, interaction, recorder, idx, recording_path)
        """
        from fnmatch import fnmatch

        # auto-expand recorder prefix → "prefix*"
        if recorder is not None and not any(c in recorder for c in "*?"):
            recorder = recorder + "*"

        # ------------------------------------------------------------------
        # 2) DEFAULT: if interaction_index is None ➜ treat it as "*"
        if interaction_index is None:
            interaction_index = "*"

        # still allow wild-cards in the string you pass
        if interaction_index and not any(c in interaction_index for c in "*?"):
            interaction_index = interaction_index + "*"

        results = []

        for loc, inter, rec, rec_dir in self.iter_recorders():
            if location    is not None and loc  != location:
                continue
            if interaction is not None and inter != interaction:
                continue
            if recorder is not None and not fnmatch(rec, recorder):
                continue

            # ── no idx filter → original behaviour ─────────────────────────
            if interaction_index is None:
                results.append((loc, inter, rec, rec_dir))
                continue

            # ── idx filter: scan immediate children of the recorder dir ────
            for entry in rec_dir.iterdir():
                idx = self._parse_interaction_index(entry.name)
                if idx and fnmatch(idx, interaction_index):
                    # BEFORE: results.append((loc, inter, rec, idx, entry))
                    # AFTER : return the recorder dir instead of the file
                    results.append((loc, inter, rec, idx, rec_dir))
                    break

        return results
    
    # ── flexible VRS finder ---------------------------------------------------
    def vrs_files(
        self,
        location:    str | None = None,
        interaction: str | None = None,
        recorder:    str | None = None,
    ):
        """
        Return a flat list of Path objects for all *.vrs files that belong to
        recorders matching the optional filters.

        Parameters
        ----------
        location, interaction, recorder
            Same semantics as `query()`:
            • None  → no filter
            • recorder may include wildcards, e.g. "iphone_*".
        """
        files = []
        for _, _, _, _, rec_dir in self.query(location, interaction, recorder):
            files.extend(self.vrs_under_recorder(rec_dir))
        return files
    
# ── demo ------------------------------------------------------------------
if __name__ == "__main__":
    idx = RecordingIndex("/data/ikea_recordings/raw")

    # print("All recorder dirs (first 8 shown):")
    # for n, (loc, inter, rec, p) in enumerate(idx.iter_recorders(), 1):
    #     print(f"{loc:10s} | {inter:7s} | {rec:15s} | {p}")
        # if n == 8:
        #     break

    # print("\nLocations:", idx.locations)

    # # Example: list .vrs files below one recorder dir
    # for _, _, _, rec_path in idx.iter_recorders():
    #     if "gripper" in rec_path.parts:
    #         print("\nVRS under", rec_path)
    #         print(idx.vrs_under_recorder(rec_path)[:5])
    #         break
    bedroom = idx.query(location="bedroom_1")
    vrs = idx.vrs_files(location="bedroom_1", interaction="gripper", recorder="aria_gripper")

    a = 2
