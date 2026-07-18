AP FLOOD EWS — 24-HOUR AUTO UPDATE
==================================

The Dash page already reloads ap_district_risk.json every 30 seconds.
This scheduled task updates the underlying real data once every 24 hours.

Each run performs:
1. Sentinel-1 fetch for 26 districts
2. CNN prediction
3. 72-hour weather fetch
4. LSTM prediction
5. Freshness-aware fusion
6. District JSON/CSV/GeoJSON aggregation
7. Risk-map generation
8. Safe alert evaluation

INSTALL
-------
Copy all files into:
C:\Users\user\Desktop\flood_monitor_clean

Open PowerShell:

cd C:\Users\user\Desktop\flood_monitor_clean

Install at 06:00 every day:

powershell -ExecutionPolicy Bypass `
    -File .\install_daily_refresh_task.ps1 `
    -Time "06:00"

Use another time by changing 06:00, for example "18:30".

RUN ONCE NOW
------------
powershell -ExecutionPolicy Bypass `
    -File .\run_daily_refresh_now.ps1

CHECK
-----
powershell -ExecutionPolicy Bypass `
    -File .\check_daily_refresh_task.ps1

REMOVE
------
powershell -ExecutionPolicy Bypass `
    -File .\remove_daily_refresh_task.ps1

LOG
---
outputs\reports\daily_refresh_task.log

IMPORTANT
---------
- The computer must be powered on.
- Internet access is required.
- Google Earth Engine authentication must remain valid.
- The dashboard does not need to be open for the scheduled task.
- When the dashboard is open, it reads the new JSON within about 30 seconds.
- SMS remains MOCK unless the saved SMS configuration is deliberately changed.
