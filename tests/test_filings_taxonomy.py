"""Tests for the filing/announcement taxonomy (plan C4): NSE/BSE announcement
classification and SEC 8-K/10-K/10-Q material-event mapping."""
from __future__ import annotations

import pytest

from data_ingestion.fundamental_feeds.nse_bse_watcher import announcement_type
from data_ingestion.fundamental_feeds.sec_edgar_watcher import material_event


# ── NSE/BSE announcement classification ──────────────────────────────────
@pytest.mark.parametrize("title,expected", [
    ("Board recommends final dividend of Rs 10 per share", "DIVIDEND"),
    ("Unaudited financial results for quarter ended June 30", "RESULTS"),
    ("Announcement under Regulation 30 - Buyback of shares", "BUYBACK"),
    ("Issue of bonus shares in ratio 1:1", "BONUS"),
    ("Intimation of QIP issue to raise funds", "FUNDRAISE"),
    ("Scheme of amalgamation of subsidiary", "M&A"),
    ("Pledge of shares by promoter", "INSIDER"),
    ("Appointment of statutory auditor", "AUDIT"),
    ("CSR policy disclosure", "CORPORATE_GOVERNANCE"),
    ("Trading window closed for quarterly results", "RESULTS"),
    ("routine compliance filing", "OTHER"),
])
def test_announcement_type(title, expected):
    assert announcement_type(title) == expected


def test_announcement_type_uses_body_too():
    # class derives from body when the title is generic
    assert announcement_type("Announcement", body="declares rights issue") == "RIGHTS"
    assert announcement_type("", body="Interim dividend declared") == "DIVIDEND"


# ── SEC filing material-event taxonomy ───────────────────────────────────
@pytest.mark.parametrize("ftype,title,expected", [
    ("10-K", "Annual report", "ANNUAL_REPORT"),
    ("10-Q", "Quarterly report", "QUARTERLY_REPORT"),
    ("10-K/A", "", "OTHER"),
    ("8-K", "8-K - Results of Operations and Financial Condition Item 2.02", "EARNINGS"),
    ("8-K", "8-K - Item 2.01 Acquisition of business", "ACQUISITION"),
    ("8-K", "8-K - Item 5.02 departure of principal officer", "MANAGEMENT_CHANGE"),
    ("8-K", "8-K - Item 4.01 change in certifying accountant", "AUDITOR_CHANGE"),
    ("8-K", "8-K - Item 7.01 regulation FD disclosure", "DISCLOSURE"),
    ("8-K", "8-K - no recognizable item number", "8K_EVENT"),
])
def test_material_event(ftype, title, expected):
    assert material_event(ftype, title) == expected


def test_material_event_case_insensitive_item():
    assert material_event("8-K", "Current Report on Form 8-K ITEM 2.02") == "EARNINGS"
