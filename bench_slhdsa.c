#include "device_profile.h"
#include "bench_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <oqs/oqs.h>

#define ALG_NAME  "SPHINCS+-SHA2-128s-simple"
#define ALG_SHORT "SLH-DSA"

void run_bench_slhdsa(BenchResult *out_results, int trials)
{
    printf("\n----------------------------------------------\n");
    printf("  Algorithm  : SLH-DSA-128s (FIPS 205)\n");
    printf("  liboqs id  : %s\n", ALG_NAME);
    printf("  Device     : %s\n", DEVICE_NAME);
    printf("  Scenario   : Per-packet signature (60 BPM = 60 ops/min)\n");
    printf("  Payload    : %d bytes per heartbeat packet\n", HEARTBEAT_PAYLOAD_BYTES);
    printf("  Note       : Hash-based - signing is slow by design\n");
    printf("  [Warning] pqm4 reports SLH-DSA-128s sign ~14 s on Cortex-M4 @ 64 MHz.\n");
    printf("  Trials     : %d\n", trials);
    printf("----------------------------------------------\n");

    OQS_SIG *sig = OQS_SIG_new(ALG_NAME);
    if (!sig) {
        fprintf(stderr,
            "[bench_slhdsa] Failed to init %s\n"
            "  Ensure liboqs was built with SPHINCS+ enabled.\n", ALG_NAME);
        for (int i = 0; i < 3; i++) {
            out_results[i].algorithm       = ALG_SHORT;
            out_results[i].pass_overall    = 0;
        }
        return;
    }

    uint8_t *pk        = malloc(sig->length_public_key);
    uint8_t *sk        = malloc(sig->length_secret_key);
    uint8_t *signature = malloc(sig->length_signature);
    size_t   sig_len;

    uint8_t heartbeat_pkt[HEARTBEAT_PAYLOAD_BYTES];
    OQS_randombytes(heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES);

    if (!pk || !sk || !signature) {
        fprintf(stderr, "[bench_slhdsa] malloc failed\n");
        goto cleanup;
    }

    size_t total_ram = sig->length_public_key
                     + sig->length_secret_key
                     + sig->length_signature;

    printf("  Key sizes: PK=%zu B  SK=%zu B  Sig=%zu B\n",
           sig->length_public_key,
           sig->length_secret_key,
           sig->length_signature);

    printf("  [Warming up - SLH-DSA keygen may take a moment...]\n");
    OQS_SIG_keypair(sig, pk, sk);
    OQS_SIG_sign(sig, signature, &sig_len,
                 heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES, sk);

    int kg_trials = (trials > 50) ? 50 : trials;
    printf("  [KeyGen: running %d trials (capped for SLH-DSA)]\n", kg_trials);

    double t_start = now_us();
    for (int i = 0; i < kg_trials; i++)
        OQS_SIG_keypair(sig, pk, sk);
    double t_keygen_us = (now_us() - t_start) / kg_trials;

    BenchResult r_keygen = {
        .algorithm       = ALG_SHORT,
        .operation       = "KeyGen",
        .pk_bytes        = sig->length_public_key,
        .sk_bytes        = sig->length_secret_key,
        .ct_or_sig_bytes = 0,
        .total_ram_bytes = total_ram
    };
    fill_timing(&r_keygen, t_keygen_us);
    evaluate_constraints(&r_keygen, BUDGET_KEYGEN_MS, BUDGET_KEY_BYTES);
    print_result(&r_keygen);
    out_results[0] = r_keygen;

    OQS_SIG_keypair(sig, pk, sk);
    int sign_trials = (trials > 100) ? 100 : trials;
    printf("  [Sign: running %d trials]\n", sign_trials);

    t_start = now_us();
    for (int i = 0; i < sign_trials; i++) {
        heartbeat_pkt[0] = (uint8_t)i;
        OQS_SIG_sign(sig, signature, &sig_len,
                     heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES, sk);
    }
    double t_sign_us = (now_us() - t_start) / sign_trials;

    BenchResult r_sign = {
        .algorithm       = ALG_SHORT,
        .operation       = "Sign",
        .pk_bytes        = 0,
        .sk_bytes        = sig->length_secret_key,
        .ct_or_sig_bytes = sig->length_signature,
        .total_ram_bytes = sig->length_secret_key + sig->length_signature
    };
    fill_timing(&r_sign, t_sign_us);
    evaluate_constraints(&r_sign, BUDGET_SIGN_MS, BUDGET_SIG_BYTES);
    print_result(&r_sign);
    out_results[1] = r_sign;

    t_start = now_us();
    for (int i = 0; i < trials; i++) {
        OQS_SIG_verify(sig, heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES,
                       signature, sig_len, pk);
    }
    double t_verify_us = (now_us() - t_start) / trials;

    BenchResult r_verify = {
        .algorithm       = ALG_SHORT,
        .operation       = "Verify",
        .pk_bytes        = sig->length_public_key,
        .sk_bytes        = 0,
        .ct_or_sig_bytes = sig->length_signature,
        .total_ram_bytes = sig->length_public_key + sig->length_signature
    };
    fill_timing(&r_verify, t_verify_us);
    evaluate_constraints(&r_verify, BUDGET_VERIFY_MS, BUDGET_SIG_BYTES);
    print_result(&r_verify);
    out_results[2] = r_verify;

    OQS_STATUS vrc = OQS_SIG_verify(sig, heartbeat_pkt,
                                    HEARTBEAT_PAYLOAD_BYTES,
                                    signature, sig_len, pk);
        printf("\n  [%s] Signature verification on heartbeat packet.\n",
            vrc == OQS_SUCCESS ? "OK" : "FAIL");

    double sig_overhead_pct = (double)sig->length_signature
                            / (double)HEARTBEAT_PAYLOAD_BYTES * 100.0;
    double uplink_bps       = (double)(sig->length_signature
                            + HEARTBEAT_PAYLOAD_BYTES) * 8.0 * 60.0;

        printf("\nBandwidth and overhead analysis\n");
    printf("  Payload size         : %d B\n",     HEARTBEAT_PAYLOAD_BYTES);
    printf("  Signature size       : %zu B\n",    sig->length_signature);
    printf("  Signature overhead   : %.1f%% of payload\n", sig_overhead_pct);
    printf("  Required uplink @ 60BPM: %.0f bps\n", uplink_bps);
    printf("  BLE 4.2 max throughput : ~125,000 bps\n");
    printf("  BLE feasible?        : %s\n",
            uplink_bps <= 125000.0 ? "YES" : "NO - exceeds BLE bandwidth");
        printf("\n");

cleanup:
    free(pk); free(sk); free(signature);
    OQS_SIG_free(sig);
}
