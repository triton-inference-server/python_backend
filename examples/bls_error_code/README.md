# BLS Error Code Propagation Test

Verifies that error codes are preserved when Python models are chained via
BLS (Business Logic Scripting).

## The Problem

Prior to the fix, when Model A returned an `InferenceResponse` with a specific
error code (e.g. `NOT_FOUND`, `UNSUPPORTED`), Model B calling it via BLS would
always see `INTERNAL` regardless of the original code.

See [triton-inference-server/server#7804](https://github.com/triton-inference-server/server/issues/7804).

## Models

- **error_source** — accepts an `ERROR_CODE` int and returns an
  `InferenceResponse` with that specific `TritonError` code.
- **bls_error_caller** — calls `error_source` via BLS, reads the error code
  from the response, and returns it as `RECEIVED_CODE`.

## Running

```bash
# Start Triton with the test models as the model repository
tritonserver --model-repository=examples/bls_error_code

# In another terminal, run the test (requires tritonclient[http])
pip install tritonclient[http]
python examples/bls_error_code/test_error_code_propagation.py
```

## Expected Output

```
  Sent: NOT_FOUND       (2) -> Got: NOT_FOUND       (2)  [OK]
  Sent: INVALID_ARG     (3) -> Got: INVALID_ARG     (3)  [OK]
  Sent: UNAVAILABLE     (4) -> Got: UNAVAILABLE     (4)  [OK]
  Sent: UNSUPPORTED     (5) -> Got: UNSUPPORTED     (5)  [OK]
  Sent: CANCELLED       (7) -> Got: CANCELLED       (7)  [OK]
ALL 5 TESTS PASSED -- error codes preserved through BLS
```
