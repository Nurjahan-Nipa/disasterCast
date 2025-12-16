#!/usr/bin/env python3
"""
dataset.py

Dataset for LLM + Temporal Encoder + H3 for POI visit forecasting.

UPDATED:
- Removed redundant visit sequence from prompt - temporal encoder handles that.
- Added metadata fields (location_name, business_category, city, placekey).
- Added seq_raw (prev 13/20 values) so evaluate.py can write them into the final CSV.

Assumptions (GLOBAL TIMELINE, no sliding window):
- Global time series starts on 19 September 2022 (day 1).
- Hurricane Ian made landfall on 28 September 2022 (day 10).
- For d14:
    * Input sequence: days 1–13  (19 Sept – 1 Oct)
    * Target: day 14            (2 Oct)  = 4 days after landfall
    * Semantics: early / partial recovery
- For d21:
    * Input sequence: days 1–20 (19 Sept – 8 Oct)
    * Target: day 21            (9 Oct)  = 11 days after landfall
    * Semantics: later recovery, near-normal activity

This file:
- Loads JSONL files
- Creates hard prompts with POI metadata + task instruction (NO visit numbers)
- Returns per item:
      prompt (string) - semantic context for LLM
      sequences (Tensor: T × 1, float32) - numerical patterns for temporal encoder
      target (float)
      h3_id (long)
      location_name (str)
      business_category (str)
      city (str)
      placekey (str)
      seq_raw (list[float]) - raw prev 13/20 values
"""

import json
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------
# Utility
# --------------------------------------------------------------

def normalize_none(x):
    """Normalize various 'none' markers to real None."""
    return x if x not in ("", None, "null", "None") else None


# --------------------------------------------------------------
# Main dataset
# --------------------------------------------------------------

class VisitForecastDataset(Dataset):
    def __init__(self, jsonl_path, horizon="d14"):
        """
        Parameters
        ----------
        jsonl_path : str
            Path to a prepared JSONL file.
        horizon : str
            'd14' or 'd21' (matches your training / evaluation scripts).
        """
        self.records = []
        self.horizon = horizon

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # ---------------------------
        # Build H3 vocab (single resolution)
        # ---------------------------
        self.h3_to_idx = {}
        for rec in self.records:
            h = normalize_none(rec.get("h3_7"))
            if h is not None and h not in self.h3_to_idx:
                # 0 is reserved for unknown / padding
                self.h3_to_idx[h] = len(self.h3_to_idx) + 1

        self.num_h3_cells = len(self.h3_to_idx) + 1

    # ----------------------------------------------------------
    # Create the LLM prompt (NO visit sequence!)
    # ----------------------------------------------------------
    def make_prompt(self, rec):
        """
        Hard prompt focusing on POI semantics and hurricane context.
        
        NO visit sequence inside the prompt – temporal encoder handles numbers.
        This prompt provides:
        - POI metadata (what the LLM understands semantically)
        - Hurricane context (dates, impact, recovery phase)
        - Task instruction
        """

        # --- POI metadata ---
        location_name = rec.get("location_name", "Unknown location")
        business_category = rec.get("business_category", "Unknown category")
        city = rec.get("city", "Unknown city")

        # --- Horizon-specific context ---
        if self.horizon == "d14":
            target_day_index = 14

            analysis_window = "13 days (from 19 September through 1 October 2022)"
            analysis_period = (
                "This period includes 9 days before landfall, the hurricane impact on "
                "28 September, and the first 3 days of initial recovery."
            )
            target_desc = (
                "4 days after Hurricane Ian's landfall, during the early recovery phase "
                "when activity is only partially returning toward normal levels"
            )
            target_date = "2 October 2022"

        else:  # horizon == "d21"
            target_day_index = 21

            analysis_window = "20 days (from 19 September through 8 October 2022)"
            analysis_period = (
                "This period includes 9 days before landfall, the hurricane impact on "
                "28 September, and the first 10 days of recovery."
            )
            target_desc = (
                "11 days after Hurricane Ian's landfall, when many places are "
                "approaching their normal visit levels again"
            )
            target_date = "9 October 2022"

        # ----------------------------- Prompt -----------------------------
        prompt = f"""You are an expert in human mobility patterns and disaster recovery dynamics.

Hurricane Ian made landfall in Florida on 28 September 2022, causing widespread disruption to normal business activity across the region.

POI Information:
- Name: {location_name}
- Business type: {business_category}
- City: {city}

Analysis Period:
You have access to visitor count patterns for this location over {analysis_window}. {analysis_period}

The temporal patterns reveal how this location's activity responded to the hurricane through the pre-impact baseline, the immediate disruption, and early recovery dynamics.

Task:
Based on the temporal patterns you can observe and the typical recovery behavior for this type of business, predict the visitor count for day {target_day_index} ({target_date}), which is {target_desc}.

Consider:
- How this type of business ({business_category}) typically recovers after major disruptions
- The location's role in the community and essential service needs
- Recovery patterns that may vary by business type and local conditions
- The stage of recovery (early vs. later phase)

Return only one non-negative integer representing the predicted visitor count.

Prediction:"""

        return prompt.strip()

    # ----------------------------------------------------------
    # Get one record
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        rec = self.records[idx]

        prompt = self.make_prompt(rec)

        # Temporal sequence + target (raw counts)
        if self.horizon == "d14":
            seq = rec["prev_13_values_raw"]
            target = rec["y_true_d14_raw"]
        else:
            seq = rec["prev_20_values_raw"]
            target = rec["y_true_d21_raw"]

        # Save raw sequence (for CSV / analysis)
        seq_raw = list(seq)

        # (T, 1) float32 for the temporal encoder
        seq_tensor = torch.tensor(seq, dtype=torch.float32).view(-1, 1)

        # H3 (single-resolution, used as separate embedding)
        h3_raw = normalize_none(rec.get("h3_7"))
        h3_id = self.h3_to_idx.get(h3_raw, 0)

        # Metadata for categorical error analysis and top-k inspection
        location_name = rec.get("location_name", "Unknown location")
        business_category = rec.get("business_category", "Unknown category")
        city = rec.get("city", "Unknown city")
        placekey = rec.get("placekey", "unknown_placekey")

        return {
            "prompt": prompt,
            "sequences": seq_tensor,   # (T, 1)
            "target": float(target),
            "h3_id": h3_id,

            # extra metadata
            "location_name": location_name,
            "business_category": business_category,
            "city": city,
            "placekey": placekey,

            # raw temporal sequence (prev 13 or 20 values)
            "seq_raw": seq_raw,
        }

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.records)


# --------------------------------------------------------------
# Collate function
# --------------------------------------------------------------

def collate_fn(batch):
    """
    Batch fields:

    prompts: list[str]
    sequences: stacked -> (B × T × 1)
    targets: (B,)
    h3_ids: (B,)

    plus metadata lists of length B and seq_raw list-of-lists.
    """
    prompts = [b["prompt"] for b in batch]

    seqs = torch.stack([b["sequences"] for b in batch], dim=0)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
    h3_ids = torch.tensor([b["h3_id"] for b in batch], dtype=torch.long)

    # Metadata
    location_names = [b["location_name"] for b in batch]
    business_categories = [b["business_category"] for b in batch]
    cities = [b["city"] for b in batch]
    placekeys = [b["placekey"] for b in batch]

    # Raw sequences (list-of-lists, keep as Python lists)
    seq_raws = [b["seq_raw"] for b in batch]

    return {
        "prompts": prompts,
        "sequences": seqs,
        "targets": targets,
        "h3_ids": h3_ids,

        "location_name": location_names,
        "business_category": business_categories,
        "city": cities,
        "placekey": placekeys,

        "seq_raw": seq_raws,
    }
