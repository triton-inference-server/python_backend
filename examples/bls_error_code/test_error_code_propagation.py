#!/usr/bin/env python3
"""Integration test: BLS error code propagation.

Verifies that when a Python model returns an InferenceResponse with a
specific TritonError code, a BLS caller model receives the same code
(not a hardcoded INTERNAL).

Usage:
    # Start Triton with the test models:
    tritonserver --model-repository=examples/bls_error_code

    # Run this test:
    python examples/bls_error_code/test_error_code_propagation.py
"""

import sys
import time

import numpy as np
import tritonclient.http as httpclient

CODE_NAMES = {
    0: "UNKNOWN",
    1: "INTERNAL",
    2: "NOT_FOUND",
    3: "INVALID_ARG",
    4: "UNAVAILABLE",
    5: "UNSUPPORTED",
    6: "ALREADY_EXISTS",
    7: "CANCELLED",
}


def wait_for_server(url="localhost:8000", timeout=60):
    client = httpclient.InferenceServerClient(url)
    start = time.time()
    while time.time() - start < timeout:
        try:
            if client.is_server_ready():
                return client
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server not ready after {timeout}s")


def test_error_code(client, send_code):
    inputs = [httpclient.InferInput("ERROR_CODE", [1], "INT32")]
    inputs[0].set_data_from_numpy(np.array([send_code], dtype=np.int32))
    outputs = [
        httpclient.InferRequestedOutput("RECEIVED_CODE"),
        httpclient.InferRequestedOutput("ERROR_MESSAGE"),
    ]
    result = client.infer("bls_error_caller", inputs, outputs=outputs)
    received = result.as_numpy("RECEIVED_CODE")[0]
    msg = result.as_numpy("ERROR_MESSAGE")[0]
    if isinstance(msg, bytes):
        msg = msg.decode()
    return received, msg


def main():
    print("Waiting for Triton server...")
    client = wait_for_server()
    print("Server ready!\n")

    print("=" * 70)
    print("BLS Error Code Propagation - Integration Test")
    print("=" * 70)

    # Test codes that are NOT the default INTERNAL (1), to catch regressions.
    test_codes = [2, 3, 4, 5, 7]  # NOT_FOUND, INVALID_ARG, UNAVAILABLE, UNSUPPORTED, CANCELLED
    failures = 0

    for code in test_codes:
        received, msg = test_error_code(client, code)
        sent_name = CODE_NAMES.get(code, f"?{code}")
        recv_name = CODE_NAMES.get(received, f"?{received}")
        status = "OK" if received == code else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"  Sent: {sent_name:15s} ({code}) -> "
            f"Got: {recv_name:15s} ({received})  [{status}]"
        )

    print()
    print("=" * 70)
    if failures == 0:
        print(f"ALL {len(test_codes)} TESTS PASSED -- error codes preserved through BLS")
    else:
        print(f"{failures}/{len(test_codes)} TESTS FAILED -- error codes lost")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
