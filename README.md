# PQC Heartbeat IoT Benchmark - Nordic nRF52840

## Overview

This README summarizes the benchmark output from `cmake-build-debug/pqc_fault_analysis` for NIST post-quantum algorithms in a heartbeat IoT scenario.

## Device and scenario

- Device: Nordic nRF52840 (Cortex-M4F @ 64 MHz)
- Scenario: Cardiac patch transmitting heartbeat data over BLE to a hospital gateway
- Heart rate: 60 BPM (1 packet/sec at rest)
- Payload size: 71 bytes per heartbeat packet, signed per transmission
- Session key establishment: once every 60 min

## Standards and algorithms

- ML-KEM-768 (NIST FIPS 203)
- ML-DSA-65 (NIST FIPS 204)
- SLH-DSA-128s, SPHINCS+-SHA2-128s-simple (NIST FIPS 205)

## Benchmark run details

- Banner: Simulated ARM Cortex-M4 @ 80 MHz, 96 KB SRAM
- Trials: 1000 per algorithm
- SLH-DSA KeyGen capped at 50 trials; Sign capped at 100 trials
- Energy budget per op: 3500 uJ
- Crypto RAM budget: 128 KB

## Results (May 15, 2026)

### Summary table

| Algorithm | Op     | Dev ms   | uJ       | Sig/CT B | RAM KB | Feasible |
| --------- | ------ | -------- | -------- | -------- | ------ | -------- |
| ML-KEM    | KeyGen | 6.49     | 93.5     | 0        | 4      | YES      |
| ML-KEM    | Encaps | 4.61     | 66.3     | 1088     | 2      | YES      |
| ML-KEM    | Decaps | 4.49     | 64.6     | 1088     | 3      | YES      |
| ML-DSA    | KeyGen | 20.49    | 295.1    | 0        | 9      | YES      |
| ML-DSA    | Sign   | 78.45    | 1129.6   | 3309     | 7      | YES      |
| ML-DSA    | Verify | 18.63    | 268.3    | 3309     | 5      | YES      |
| SLH-DSA   | KeyGen | 7978.53  | 114890.9 | 0        | 7      | NO       |
| SLH-DSA   | Sign   | 61380.96 | 883885.8 | 7856     | 7      | NO       |
| SLH-DSA   | Verify | 58.54    | 843.0    | 7856     | 7      | NO       |

### ML-KEM-768 details

- Key sizes: PK 1184 B, SK 2400 B, CT 1088 B
- Session energy (KeyGen + Encaps + Decaps): 224.42 uJ
- Battery drain per hour: 0.0000% of CR2032

### ML-DSA-65 details

- Key sizes: PK 1952 B, SK 4032 B, Sig 3309 B
- 60 BPM projection: Sign energy/hr 4066705.44 uJ, Verify energy/hr 965908.80 uJ
- 60 BPM projection: Total crypto energy/hr 5032614.24 uJ, Battery drain/hr 0.1983% of CR2032
- 60 BPM projection: Bandwidth sig=3309 B x 60 BPM = 3309.0 B/s uplink

### SLH-DSA-128s details

- Key sizes: PK 32 B, SK 64 B, Sig 7856 B
- Warning: pqm4 reports SLH-DSA-128s sign ~14 s on Cortex-M4 @ 64 MHz
- Bandwidth and overhead: Signature overhead 11064.8% of payload
- Bandwidth and overhead: Required uplink @ 60 BPM 63416 bps
- Bandwidth and overhead: BLE 4.2 max throughput ~125,000 bps, BLE feasible NO
- SLH-DSA Verify fails memory constraint only (signature size); timing and energy are within budget

## Notes from the benchmark output

- Device time is a lower-bound estimate (host-scaled).
- Real Cortex-M4 without FPU optimization: ~5-10x slower.
- SLH-DSA signing may exceed budgets on real HW.

## Recommendation from the benchmark output

- Session key exchange -> ML-KEM-768 (FIPS 203). Replaces RSA/ECDH. Encaps/Decaps within budget. Run once/hour.
- Per-packet signing -> ML-DSA-65 (FIPS 204). Fastest lattice signer. Signature ~3.3 KB. Verify fast enough.
- Alternative signing -> SLH-DSA-128s (FIPS 205). Conservative security (hash-only). Sign latency 61381 ms on device (200x over 300 ms budget). Best for infrequent ops (session auth, firmware signing), not per-packet.
