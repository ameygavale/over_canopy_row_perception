# GPS / RTK Correction Pipeline

## Hardware

- **Base station**: Emlid RS3
- **SIM**: Mint Mobile (APN: fast.t-mobile.com)
- **Caster**: Emlid Caster (caster.emlid.com:2101)
  - Mountpoint: MP24398
  - Password: 342yqt
  - Rover user: u54947 / 943she

## Fallback

- u-blox PointPerfect Flex over Starlink backhaul

## ROS2 Arbitration Node (planned)

Two NTRIP client nodes → arbitration node monitors fix quality at 1Hz.
Prefers base station, falls back to PointPerfect on timeout or sustained float.

## Open Issues

- [ ] RELPOSNED → NavSatFix converter node not yet added
- [ ] GNSS correction input via Amiga gRPC bridge not yet resolved
