"""
Incremental (batch) processing.

A batch is a directory with any subset of:

    customers.csv
    transactions.csv
    interactions.csv
    documents.csv

Flow per batch:

    existing processed data + new raw records
        -> schema check
        -> validation / referential integrity
        -> deduplication (within batch and against existing)
        -> cleaning (same transform_* functions as full load)
        -> upsert by primary key
        -> staged + atomic promotion

Every applied batch is recorded in a ledger with a content
fingerprint, so re-applying the same batch is a no-op.

Usage:
    python -m src.data_engineering.incremental data/incoming/batch_001
"""

import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_engineering.cleaner import (
    DATASET_ORDER,
    PROCESSED_DIR,
    QUARANTINE_DIR,
    SCHEMAS,
    TRANSFORMS,
    check_schema,
    configure_logging,
    discard_staging,
    logger,
    processed_path,
    promote_outputs,
    transform_customers,
)


LEDGER_FILE = "_batches.json"

DATE_COLUMNS = {
    "transactions": "transaction_date",
    "interactions": "interaction_date",
    "documents": "document_date",
}


# ==============================================================
# LEDGER
# ==============================================================

def batch_fingerprint(batch_dir: Path) -> str:

    digest = hashlib.sha256()

    for name in DATASET_ORDER:

        path = batch_dir / f"{name}.csv"

        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())

    return digest.hexdigest()


def load_ledger(processed_dir: Path) -> list[dict]:

    path = processed_dir / LEDGER_FILE

    if not path.exists():
        return []

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_ledger(
    processed_dir: Path,
    ledger: list[dict],
) -> None:

    path = processed_dir / LEDGER_FILE
    tmp_path = path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(ledger, file, indent=4)

    os.replace(tmp_path, path)


# ==============================================================
# UPSERT
# ==============================================================

def as_stored(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round-trip through CSV so freshly cleaned rows have the
    same representation as rows read back from disk (dates as
    strings etc.). Makes the "unchanged" comparison reliable.
    """

    if df.empty:
        return df.copy()

    return pd.read_csv(
        io.StringIO(df.to_csv(index=False))
    )


def upsert(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    primary_key: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Insert new keys, replace changed rows, skip identical rows.
    """

    incoming = as_stored(incoming)

    if incoming.empty:
        return existing, {
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "updated_ids": [],
        }

    columns = list(existing.columns)
    incoming = incoming[columns]

    existing_by_key = existing.set_index(primary_key)

    is_existing = incoming[primary_key].isin(
        existing_by_key.index
    )

    new_rows = incoming[~is_existing]
    candidates = incoming[is_existing]

    def row_text(df: pd.DataFrame) -> pd.Series:
        return (
            df.astype("string")
            .fillna("")
            .agg("|".join, axis=1)
        )

    changed_mask = pd.Series(False, index=candidates.index)

    if not candidates.empty:

        old = existing_by_key.loc[
            candidates[primary_key]
        ].reset_index()

        changed_mask = pd.Series(
            row_text(old[columns]).values
            != row_text(candidates[columns]).values,
            index=candidates.index,
        )

    updated_rows = candidates[changed_mask]

    merged = existing[
        ~existing[primary_key].isin(
            updated_rows[primary_key]
        )
    ]

    merged = pd.concat(
        [merged, updated_rows, new_rows],
        ignore_index=True,
    )

    return merged, {
        "inserted": int(len(new_rows)),
        "updated": int(len(updated_rows)),
        "unchanged": int((~changed_mask).sum()),
        "updated_ids": updated_rows[primary_key].tolist(),
    }


# ==============================================================
# APPLY BATCH
# ==============================================================

def apply_batch(
    batch_dir: Path,
    processed_dir: Path = PROCESSED_DIR,
    quarantine_dir: Path = QUARANTINE_DIR,
) -> dict:

    batch_dir = Path(batch_dir)

    if not batch_dir.is_dir():
        raise FileNotFoundError(
            f"Batch directory not found: {batch_dir}"
        )

    batch_id = batch_dir.name
    fingerprint = batch_fingerprint(batch_dir)
    ledger = load_ledger(processed_dir)

    # ----------------------------------------------------------
    # Idempotency: same batch content already applied.
    # ----------------------------------------------------------

    for entry in ledger:
        if entry["fingerprint"] == fingerprint:
            logger.info(
                f"Batch {batch_id} already applied at "
                f"{entry['applied_at']} - skipping."
            )
            return {
                "batch_id": batch_id,
                "status": "skipped",
                "reason": "already_applied",
            }

    run_id = (
        f"{batch_id}_"
        + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    logger.info(
        f"Applying incremental batch {batch_id}"
    )

    existing = {
        name: pd.read_csv(
            processed_path(processed_dir, name)
        )
        for name in DATASET_ORDER
    }

    staged = {}
    quarantine = {}
    report = {
        "batch_id": batch_id,
        "status": "applied",
        "datasets": {},
    }

    try:

        # ------------------------------------------------------
        # Customers first - they are the reference for every
        # other dataset in the same batch.
        # ------------------------------------------------------

        customers = existing["customers"]

        customer_file = batch_dir / "customers.csv"

        if customer_file.exists():

            raw = check_schema(
                pd.read_csv(customer_file),
                "customers",
            )

            clean, rejected = transform_customers(raw)

            customers, stats = upsert(
                existing["customers"],
                clean,
                SCHEMAS["customers"]["primary_key"],
            )

            stats.pop("updated_ids")
            stats["received"] = int(len(raw))
            stats["quarantined"] = int(len(rejected))

            staged["customers"] = customers
            quarantine["customers"] = rejected
            report["datasets"]["customers"] = stats

        for name, transform in TRANSFORMS.items():

            batch_file = batch_dir / f"{name}.csv"

            if not batch_file.exists():
                continue

            raw = check_schema(
                pd.read_csv(batch_file),
                name,
            )

            # --------------------------------------------------
            # Referential integrity (reported explicitly before
            # cleaning quarantines the rows).
            # --------------------------------------------------

            unknown = sorted(
                set(raw["customer_id"].dropna().astype(str))
                - set(customers["customer_id"].astype(str))
            )

            if unknown:
                logger.warning(
                    f"[REFERENTIAL INTEGRITY] {name}: "
                    f"{len(unknown)} unknown customer id(s) "
                    f"{unknown[:10]} - rows will be "
                    f"quarantined."
                )

            clean, rejected = transform(
                raw,
                customers,
            )

            primary_key = SCHEMAS[name]["primary_key"]

            # --------------------------------------------------
            # Late-arriving records: NEW keys dated before the
            # newest record we already hold. They are accepted,
            # but feature snapshots on/after that date are now
            # stale - and so are snapshots after any corrected
            # (updated) record.
            # --------------------------------------------------

            date_column = DATE_COLUMNS[name]

            latest_existing = pd.to_datetime(
                existing[name][date_column],
                errors="coerce",
            ).max()

            is_new_key = ~clean[primary_key].isin(
                existing[name][primary_key]
            )

            late = clean[
                is_new_key
                & (clean[date_column] < latest_existing)
            ]

            merged, stats = upsert(
                existing[name],
                clean,
                primary_key,
            )

            updated_ids = set(stats.pop("updated_ids"))

            stale = pd.concat(
                [
                    late,
                    clean[clean[primary_key].isin(updated_ids)],
                ]
            )

            stats["received"] = int(len(raw))
            stats["quarantined"] = int(len(rejected))
            stats["duplicates_in_batch"] = int(
                len(raw) - len(rejected) - len(clean)
            )
            stats["late_arriving"] = int(len(late))

            if not late.empty:
                logger.warning(
                    f"[LATE DATA] {name}: {len(late)} new "
                    f"record(s) dated before "
                    f"{latest_existing.date()}."
                )

            if not stale.empty:
                stats["recompute_features_from"] = str(
                    stale[date_column].min().date()
                )
                stats["affected_customers"] = sorted(
                    stale["customer_id"].astype(str).unique()
                )

                logger.warning(
                    f"[STALE FEATURES] {name}: snapshots from "
                    f"{stats['recompute_features_from']} "
                    f"onwards for "
                    f"{len(stats['affected_customers'])} "
                    f"customer(s) need to be rebuilt."
                )

            staged[name] = merged
            quarantine[name] = rejected
            report["datasets"][name] = stats

            logger.info(
                f"{name}: {stats}"
            )

        # ------------------------------------------------------
        # Quarantine is appended (not replaced) in incremental
        # mode so earlier rejected rows are kept for review.
        # ------------------------------------------------------

        for name, rejected in list(quarantine.items()):

            path = (
                quarantine_dir
                / f"{name}_quarantine.csv"
            )

            if path.exists():
                previous = pd.read_csv(path)
                rejected = pd.concat(
                    [previous, as_stored(rejected)],
                    ignore_index=True,
                ).drop_duplicates()

            quarantine[name] = rejected

        promote_outputs(
            staged,
            quarantine,
            processed_dir,
            quarantine_dir,
            run_id,
            run_type="incremental",
        )

    except Exception:

        discard_staging(processed_dir, run_id)

        logger.exception(
            f"Batch {batch_id} FAILED. Nothing was "
            f"promoted; the batch can be re-applied after "
            f"the cause is fixed."
        )

        raise

    ledger.append(
        {
            "batch_id": batch_id,
            "fingerprint": fingerprint,
            "applied_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "datasets": report["datasets"],
        }
    )

    save_ledger(processed_dir, ledger)

    logger.info(
        f"Batch {batch_id} applied"
    )

    return report


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    configure_logging()

    if len(sys.argv) < 2:
        print(
            "Usage: python -m "
            "src.data_engineering.incremental <batch_dir>"
        )
        raise SystemExit(2)

    report = apply_batch(Path(sys.argv[1]))

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
