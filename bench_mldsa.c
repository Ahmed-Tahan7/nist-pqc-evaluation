#include "device_profile.h"
#include "bench_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <oqs/oqs.h>

#define ALG_NAME  "ML-DSA-65"
#define ALG_SHORT "ML-DSA"

void run_bench_mldsa(BenchResult *out_results, int trials) {
    printf("\n----------------------------------------------\n");
    printf("  Algorithm  : %s\n", ALG_NAME);
    printf("  Device     : %s\n", DEVICE_NAME);
    printf("  Scenario   : Per-packet signature (60 BPM = 60 ops/min)\n");
    printf("  Payload    : %d bytes per heartbeat packet\n", HEARTBEAT_PAYLOAD_BYTES);
    printf("  Trials     : %d\n", trials);
    printf("----------------------------------------------\n");

    OQS_SIG *sig = OQS_SIG_new(ALG_NAME);
    if (!sig) {
        fprintf(stderr, "[bench_mldsa] Failed to init %s\n", ALG_NAME);
        return;
    }

    uint8_t *pk        = malloc(sig->length_public_key);
    uint8_t *sk        = malloc(sig->length_secret_key);
    uint8_t *signature = malloc(sig->length_signature);
    size_t   sig_len;

    uint8_t heartbeat_pkt[HEARTBEAT_PAYLOAD_BYTES];
    OQS_randombytes(heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES);

    if (!pk || !sk || !signature) {
        fprintf(stderr, "[bench_mldsa] malloc failed\n");
        goto cleanup;
    }

    size_t total_ram = sig->length_public_key
                     + sig->length_secret_key
                     + sig->length_signature;

    OQS_SIG_keypair(sig, pk, sk);
    OQS_SIG_sign(sig, signature, &sig_len,
                 heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES, sk);

    double t_start = now_us();
    for (int i = 0; i < trials; i++)
        OQS_SIG_keypair(sig, pk, sk);
    double t_keygen_us = (now_us() - t_start) / trials;

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

    t_start = now_us();
    for (int i = 0; i < trials; i++) {
        heartbeat_pkt[0] = (uint8_t)i;
        OQS_SIG_sign(sig, signature, &sig_len,
                     heartbeat_pkt, HEARTBEAT_PAYLOAD_BYTES, sk);
    }
    double t_sign_us = (now_us() - t_start) / trials;

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

    double ops_per_min      = 60.0;
    double sign_energy_hr   = r_sign.energy_uj * ops_per_min * 60.0;
    double verify_energy_hr = r_verify.energy_uj * ops_per_min * 60.0;
    double total_energy_hr  = sign_energy_hr + verify_energy_hr;
    double drain_pct_hr     = (total_energy_hr / 1000.0) / BATTERY_CAPACITY_MJ * 100.0;

        printf("\n60 BPM operational projection\n");
    printf("  Sign ops/hr          : %.0f\n",      ops_per_min * 60.0);
        printf("  Sign energy/hr       : %.2f uJ\n",   sign_energy_hr);
        printf("  Verify energy/hr     : %.2f uJ\n",   verify_energy_hr);
        printf("  Total crypto energy/hr: %.2f uJ\n",  total_energy_hr);
    printf("  Battery drain/hr     : %.4f%% of CR2032\n", drain_pct_hr);
        printf("  Bandwidth: sig=%zu B x 60 BPM = %.1f B/s uplink\n",
           sig->length_signature,
           (double)sig->length_signature * 60.0 / 60.0);
        printf("\n");

cleanup:
    free(pk); free(sk); free(signature);
    OQS_SIG_free(sig);
}
